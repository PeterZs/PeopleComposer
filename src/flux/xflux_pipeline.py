from PIL import Image, ExifTags
import numpy as np
import torch
from torch import Tensor

from torchvision.transforms.functional import gaussian_blur
import math
from einops import rearrange
import uuid
import os
from tqdm import tqdm

from src.flux.modules.layers import (
    SingleStreamBlockProcessor,
    DoubleStreamBlockProcessor,
    SingleStreamBlockLoraProcessor,
    DoubleStreamBlockLoraProcessor,
    IPDoubleStreamBlockProcessor,
    ImageProjModel,
)
from src.flux.sampling import denoise, denoise_controlnet, get_noise, get_schedule, prepare, unpack, prepare_dual, denoise_ar, create_black_image_latents, prepare_dual_train_ar, prepare_dual_train_ar_embed_id, prepare_w_prompt_mask, prepare_dual_train_ar_pretend_pe, prepare_dual_train_ar_pretend_pe_learnable, prepare_dual_train_layout, prepare_split_resolution, prepare_dual_train_ar_embed_id_realnotadded, prepare_dual_train_ar_version2, prepare_dual_train_oneshot, prepare_vanilla_t2i, prepare_dual_train_ar_w_rewrite
from src.flux.util import (
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
    load_controlnet,
    load_flow_model_quintized,
    Annotator,
    get_lora_rank,
    load_checkpoint
)

from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor


from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from typing import List
from importlib import resources

@torch.no_grad()
def init_partial_noise_with_boxes_patchified(
    noise_tokens,
    x_1,                       # [B, seqlen, dim], VAE×8 + patch×2 tokens
    input_bounding_boxes,      # list[list[xyxy]]; bs=1 -> use input_bounding_boxes[0]
    img_size,                  # (H, W) original image size (pixels)
    already_denoised_frac=0.30,   # e.g., 0.30 => start t=0.70 inside boxes
    blur_sigma=1.0,            # gaussian blur sigma on token grid
    exclude_last=True,         # exclude the last bbox from the softened region
    seed=0,                    # seed for noise
):
    """
    Returns:
    x_t0:   [B, seqlen, dim]  initial token latent for sampling
    t0_map: [1, H_lat, W_lat, 1] float32 per-token starting t field (debug)
    Token grid size is H_lat × W_lat where:
    H_lat = ceil(H/16), W_lat = ceil(W/16)
    """
    device = x_1.device
    dtype = x_1.dtype
    B, seqlen, dim = x_1.shape
    assert B == 1, "Current implementation assumes batch size = 1"

    H, W = img_size
    H_lat = math.ceil(H / 16)  # matches your packing+patchify scheme
    W_lat = math.ceil(W / 16)
    assert seqlen == H_lat * W_lat, f"seqlen ({seqlen}) != H_lat*W_lat ({H_lat*W_lat})"

    # 1) build binary mask on token grid for all boxes except last (optional)
    mask = torch.zeros((1, 1, H_lat, W_lat), device=device, dtype=torch.float32)

    bboxes = input_bounding_boxes[0] if isinstance(input_bounding_boxes[0], (list, tuple)) and isinstance(input_bounding_boxes[0][0], (list, tuple)) else input_bounding_boxes
    # allow both [[...],[...]] and [[[...],[...]]]
    if exclude_last and len(bboxes) > 0:
        bboxes_iter = bboxes[:-1]
    else:
        bboxes_iter = bboxes

    for bbox in bboxes_iter:
        x1, y1, x2, y2 = bbox  # xyxy in original pixels
        # map to token grid coords (ceil/16 resolution); use float -> int ranges
        gx1 = int(max(0, math.floor(x1 / 16.0)))
        gy1 = int(max(0, math.floor(y1 / 16.0)))
        gx2 = int(min(W_lat, math.ceil(x2 / 16.0)))
        gy2 = int(min(H_lat, math.ceil(y2 / 16.0)))
        if gx2 > gx1 and gy2 > gy1:
            mask[:, :, gy1:gy2, gx1:gx2] = 1.0

    # 2) gaussian blur the mask to soften edges (operate in float32 to be safe)
    ksize = max(1, int(4 * blur_sigma + 1))
    if ksize % 2 == 0:
        ksize += 1  # ensure odd
    mask_soft = gaussian_blur(mask, kernel_size=(ksize, ksize), sigma=blur_sigma)
    maxv = mask_soft.max()
    if maxv > 0:
        mask_soft = mask_soft / maxv
    # mask_soft: [1,1,H_lat,W_lat] in [0,1], float32

    # 3) per-token starting t field: inside ~ (1 - frac), outside ~ 1.0
    start_t_inside = 1.0 - float(already_denoised_frac)  # e.g., 0.70 for 30%
    start_t_outside = 1.0
    t0_map = mask_soft * start_t_inside + (1.0 - mask_soft) * start_t_outside
    t0_map = t0_map.clamp(0.0, 1.0)  # [1,1,H_lat,W_lat], float32

    # # 4) sample patchified noise using your exact initializer (bf16, packed, patchified)
    # noise_tokens = get_noise_patchified(
    #     height=H, width=W, device=device, dtype=dtype, seed=seed
    # )  # [1, seqlen, dim] with dim=64 if c=16 & ph=pw=2

    # 5) mix tokens: x_t0 = (1 - t) * x_1 + t * noise
    # reshape to [B,H_lat,W_lat,dim] to apply per-token t easily
    x1_img = x_1.view(1, H_lat, W_lat, dim)
    n_img  = noise_tokens.view(1, H_lat, W_lat, dim)
    t_img  = t0_map.permute(0, 2, 3, 1).to(x_1.dtype)  # [1,H_lat,W_lat,1], same dtype as x_1
    x_t0_img = (1.0 - t_img) * x1_img + t_img * n_img
    x_t0 = x_t0_img.view(1, seqlen, dim)

    # return x_t0 in the same dtype as x_1, and t0_map for debugging
    return x_t0, t0_map.permute(0, 2, 3, 1)  # [1,H_lat,W_lat,1] float32


def segment_on_white(image_pil: Image.Image, bbox_xyxy: List[int], device) -> Image.Image:
    cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    ckpt = "<YOUR_SAM2_CHECKPOINT_PATH>"

    predictor = SAM2ImagePredictor(build_sam2(cfg, ckpt, device=device))



    # image_pil: PIL.Image; bbox_xyxy: [x_min, y_min, x_max, y_max]
    if image_pil.mode != "RGB":
        image_pil = image_pil.convert("RGB")
    image_np = np.array(image_pil)

    x1, y1, x2, y2 = map(int, bbox_xyxy)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox (xyxy): {bbox_xyxy}")

    # Expand bbox in all directions
    height, width = image_np.shape[:2]
    expand_pixels = 20  # Expand by 20 pixels in each direction

    x1 = max(0, x1 - expand_pixels)
    y1 = max(0, y1 - expand_pixels)
    x2 = min(width, x2 + expand_pixels)
    y2 = min(height, y2 + expand_pixels)

    predictor.set_image(image_np)

    box = np.array([x1, y1, x2, y2], dtype=np.float32)
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        masks, scores, _ = predictor.predict(box=box)

    mask = masks[int(np.argmax(scores))].astype(bool)

    white = np.ones_like(image_np) * 255
    white[mask] = image_np[mask]
    del predictor
    return Image.fromarray(white.astype(np.uint8))

class XFluxPipeline:
    def __init__(self, model_type, device, offload: bool = False):
        self.device = torch.device(device)
        self.offload = offload
        self.model_type = model_type

        init_device = "cpu" if self.offload else self.device
        self.clip = load_clip(init_device)
        self.t5 = load_t5(init_device, max_length=512)
        self.ae = load_ae(model_type, device="cpu" if offload else self.device)
        if "fp8" in model_type:
            self.model = load_flow_model_quintized(model_type, device="cpu" if offload else self.device)
        else:
            self.model = load_flow_model(model_type, device="cpu" if offload else self.device)

        self.image_encoder_path = "openai/clip-vit-large-patch14"
        self.hf_lora_collection = "XLabs-AI/flux-lora-collection"
        self.lora_types_to_names = {
            "realism": "lora.safetensors",
        }
        self.controlnet_loaded = False
        self.ip_loaded = False

    def set_ip(self, local_path: str = None, repo_id = None, name: str = None):
        self.model.to(self.device)

        # unpack checkpoint
        checkpoint = load_checkpoint(local_path, repo_id, name)
        prefix = "double_blocks."
        blocks = {}
        proj = {}

        for key, value in checkpoint.items():
            if key.startswith(prefix):
                blocks[key[len(prefix):].replace('.processor.', '.')] = value
            if key.startswith("ip_adapter_proj_model"):
                proj[key[len("ip_adapter_proj_model."):]] = value

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()

        # setup image embedding projection model
        self.improj = ImageProjModel(4096, 768, 4)
        self.improj.load_state_dict(proj)
        self.improj = self.improj.to(self.device, dtype=torch.bfloat16)

        ip_attn_procs = {}

        for name, _ in self.model.attn_processors.items():
            ip_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    ip_state_dict[k.replace(f'{name}.', '')] = checkpoint[k]
            if ip_state_dict:
                ip_attn_procs[name] = IPDoubleStreamBlockProcessor(4096, 3072)
                ip_attn_procs[name].load_state_dict(ip_state_dict)
                ip_attn_procs[name].to(self.device, dtype=torch.bfloat16)
            else:
                ip_attn_procs[name] = self.model.attn_processors[name]

        self.model.set_attn_processor(ip_attn_procs)
        self.ip_loaded = True

    def set_lora(self, local_path: str = None, repo_id: str = None,
                 name: str = None, lora_weight: int = 0.7):
        checkpoint = load_checkpoint(local_path, repo_id, name)
        self.update_model_with_lora(checkpoint, lora_weight)

    def set_lora_from_collection(self, lora_type: str = "realism", lora_weight: int = 0.7):
        checkpoint = load_checkpoint(
            None, self.hf_lora_collection, self.lora_types_to_names[lora_type]
        )
        self.update_model_with_lora(checkpoint, lora_weight)

    def update_model_with_lora(self, checkpoint, lora_weight):
        rank = get_lora_rank(checkpoint)
        lora_attn_procs = {}

        for name, _ in self.model.attn_processors.items():
            lora_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    lora_state_dict[k[len(name) + 1:]] = checkpoint[k] * lora_weight

            if len(lora_state_dict):
                if name.startswith("single_blocks"):
                    lora_attn_procs[name] = SingleStreamBlockLoraProcessor(dim=3072, rank=rank)
                else:
                    lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(dim=3072, rank=rank)
                lora_attn_procs[name].load_state_dict(lora_state_dict)
                lora_attn_procs[name].to(self.device)
            else:
                if name.startswith("single_blocks"):
                    lora_attn_procs[name] = SingleStreamBlockProcessor()
                else:
                    lora_attn_procs[name] = DoubleStreamBlockProcessor()

        self.model.set_attn_processor(lora_attn_procs)

    def set_controlnet(self, control_type: str, local_path: str = None, repo_id: str = None, name: str = None):
        self.model.to(self.device)
        self.controlnet = load_controlnet(self.model_type, self.device).to(torch.bfloat16)

        checkpoint = load_checkpoint(local_path, repo_id, name)
        self.controlnet.load_state_dict(checkpoint, strict=False)
        self.annotator = Annotator(control_type, self.device)
        self.controlnet_loaded = True
        self.control_type = control_type

    def get_image_proj(
        self,
        image_prompt: Tensor,
    ):
        # encode image-prompt embeds
        image_prompt = self.clip_image_processor(
            images=image_prompt,
            return_tensors="pt"
        ).pixel_values
        image_prompt = image_prompt.to(self.image_encoder.device)
        image_prompt_embeds = self.image_encoder(
            image_prompt
        ).image_embeds.to(
            device=self.device, dtype=torch.bfloat16,
        )
        # encode image
        image_proj = self.improj(image_prompt_embeds)
        return image_proj


    def create_black_image(self, size=(512, 512)): # size (width, height)
        return Image.new('RGB', size, (0, 0, 0)) # Black image

    def create_white_image(self, size=(512, 512)): # size (width, height)
        return Image.new('RGB', size, (255, 255, 255)) # Black image


    def get_condition_latents_patchified(self, width, height, background_color):
        img_pil = self.create_black_image((width, height))
        img_np = np.array(img_pil, dtype=np.float32)
        if background_color == "gray":
            black_mask = (img_np[:,:,0] == 0) & (img_np[:,:,1] == 0) & (img_np[:,:,2] == 0)
            img_np[black_mask] = [127.5, 127.5, 127.5]
        img_tensor = torch.from_numpy((img_np / 127.5) - 1) # Normalize to [-1, 1]
        img_tensor = img_tensor.permute(2, 0, 1) # HWC -> CHW

        img_tensor = img_tensor.unsqueeze(0) # (1, C, H, W)
        vae_device = next(self.ae.encoder.parameters()).device

        # Encode with VAE
        with torch.no_grad():
            x_1 = self.ae.encode(img_tensor.to(vae_device).to(torch.float32))

        x_1_patchified = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        return x_1_patchified.to(torch.bfloat16)  # [1, 1024, 64]

    def get_white_latents_patchified(self, width, height):
        img_pil = self.create_white_image((width, height))
        img_np = np.array(img_pil, dtype=np.float32)
        img_tensor = torch.from_numpy((img_np / 127.5) - 1) # Normalize to [-1, 1]
        img_tensor = img_tensor.permute(2, 0, 1) # HWC -> CHW

        img_tensor = img_tensor.unsqueeze(0) # (1, C, H, W)
        vae_device = next(self.ae.encoder.parameters()).device

        # Encode with VAE
        with torch.no_grad():
            x_1 = self.ae.encode(img_tensor.to(vae_device).to(torch.float32))

        x_1_patchified = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        return x_1_patchified.to(torch.bfloat16)  # [1, 1024, 64]



    def get_segmented_latents_patchified(self, segmented_image):
        img_pil = segmented_image
        img_np = np.array(img_pil, dtype=np.float32)

        img_tensor = torch.from_numpy((img_np / 127.5) - 1) # Normalize to [-1, 1]
        img_tensor = img_tensor.permute(2, 0, 1) # HWC -> CHW

        img_tensor = img_tensor.unsqueeze(0) # (1, C, H, W)
        vae_device = next(self.ae.encoder.parameters()).device

        # Encode with VAE
        with torch.no_grad():
            x_1 = self.ae.encode(img_tensor.to(vae_device).to(torch.float32))

        x_1_patchified = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        return x_1_patchified.to(torch.bfloat16)  # [1, 1024, 64]



    def get_noise_patchified(self, height, width, seed):
        noise = get_noise(
            1, height, width, device=self.device,
            dtype=torch.bfloat16, seed=seed
        )
        noise_patchified = rearrange(noise, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        return noise_patchified

    def prepare_multi_person_rewrite(self, round_prompt, round_rewrite_full_prompt, round_rewrite_tag, round_rewrite_segment, general_prompt, width, height, customize_img_ids, bounding_boxes_in_order, customize_txt_ids, generate_img_ids_type, embed_id=False, pretend_pe=False, pe_learnable=False, apply_smaller_pose=1, oneshot=False, t2i=False):
        print("prepare_dual_train_ar")
        inp_cond = prepare_dual_train_ar_w_rewrite(self.t5,
                        self.clip,
                        bs=1,
                        # condition
                        h_condition=[height//8//apply_smaller_pose],
                        w_condition=[width//8//apply_smaller_pose],
                        # pose
                        h_pose=[height//8//apply_smaller_pose],
                        w_pose=[width//8//apply_smaller_pose],
                        # real
                        h_real=[height//8],
                        w_real=[width//8],
                        device=self.device,
                        prompt=round_prompt, # detailed_prompt
                        rewrite_full_caption=round_rewrite_full_prompt,
                        rewrite_segment_tag=round_rewrite_tag,
                        rewrite_segment_text=round_rewrite_segment,
                        clip_prompt=general_prompt, # general_prompt
                        pose_id_value=1,
                        add_pose_clip=True,
                        bounding_boxes_in_order=bounding_boxes_in_order,
                        customize_img_ids=customize_img_ids,
                        customize_txt_ids=customize_txt_ids,
                        generate_img_ids_type=generate_img_ids_type,
                        apply_smaller_pose=apply_smaller_pose
                        )
        return inp_cond

    def prepare_multi_person(self, round_prompt, general_prompt, width, height, customize_img_ids, bounding_boxes_in_order, customize_txt_ids, generate_img_ids_type, embed_id=False, pretend_pe=False, pe_learnable=False, apply_smaller_pose=1, oneshot=False, t2i=False):
        if embed_id:
            print("prepare_dual_train_ar_embed_id")
            inp_cond = prepare_dual_train_ar_embed_id_realnotadded(self.t5,
                            self.clip,
                            bs=1,
                            # condition
                            h_condition=[height//8],
                            w_condition=[width//8],
                            # pose
                            h_pose=[height//8],
                            w_pose=[width//8],
                            # real
                            h_real=[height//8],
                            w_real=[width//8],
                            device=self.device,
                            prompt=round_prompt, # detailed_prompt
                            clip_prompt=general_prompt, # general_prompt
                            add_pose_clip=True,
                            bounding_boxes_in_order=bounding_boxes_in_order
                            )
        elif pretend_pe:
            print("prepare_dual_train_ar_pretend_pe")
            inp_cond = prepare_dual_train_ar_pretend_pe(t5=self.t5,
                                    clip=self.clip,
                                    bs=1,
                                    multi_person_last_indicator=[True],
                                    h_condition=[height//8],
                                    w_condition=[width//8],
                                    ori_center=None,
                                    new_center=None,
                                    resize_scale=None,
                                    bounding_boxes_in_order=None,
                                    # pose
                                    h_pose=[height//8],
                                    w_pose=[width//8],
                                    # real
                                    h_real=[height//8],
                                    w_real=[width//8],
                                    device=self.device,
                                    prompt=round_prompt,
                                    clip_prompt=general_prompt,
                                    # TODO always not add h now!
                                    add_pose_clip=True,
                                    sample_mode=True,
                                    )
        elif pe_learnable:
            print("prepare_dual_train_ar_pretend_pe_learnable")
            inp_cond = prepare_dual_train_ar_pretend_pe_learnable(
                                    t5=self.t5,
                                    clip=self.clip,
                                    bs=1,
                                    h_condition=[height//8],
                                    w_condition=[width//8],
                                    # pose
                                    h_pose=[height//8],
                                    w_pose=[width//8],
                                    # real
                                    h_real=[height//8],
                                    w_real=[width//8],
                                    device=self.device,
                                    prompt=round_prompt,
                                    clip_prompt=general_prompt,
                                    pose_id_value=1,
                                    add_pose_clip=True,
                                    )
        elif oneshot:
            print("prepare_dual_train_oneshot")
            inp_cond = prepare_dual_train_oneshot(self.t5,
                            self.clip,
                            bs=1,
                            # pose
                            h_pose=[height//8//apply_smaller_pose],
                            w_pose=[width//8//apply_smaller_pose],
                            # real
                            h_real=[height//8],
                            w_real=[width//8],
                            device=self.device,
                            prompt=round_prompt, # detailed_prompt
                            clip_prompt=general_prompt, # general_prompt
                            pose_id_value=1,
                            add_pose_clip=True,
                            bounding_boxes_in_order=bounding_boxes_in_order,
                            customize_img_ids=customize_img_ids,
                            customize_txt_ids=customize_txt_ids,
                            generate_img_ids_type=generate_img_ids_type,
                            apply_smaller_pose=apply_smaller_pose
                            )
        elif t2i:
            print("prepare_t2i")
            inp_cond = prepare_vanilla_t2i(self.t5,
                            self.clip,
                            bs=1,
                            # real
                            h_real=[height//8],
                            w_real=[width//8],
                            device=self.device,
                            prompt=round_prompt, # detailed_prompt
                            clip_prompt=general_prompt, # general_prompt
                            bounding_boxes_in_order=bounding_boxes_in_order,
                            customize_img_ids=customize_img_ids,
                            customize_txt_ids=customize_txt_ids
                            )

        else:
            print("prepare_dual_train_ar")
            inp_cond = prepare_dual_train_ar(self.t5,
                            self.clip,
                            bs=1,
                            # condition
                            h_condition=[height//8//apply_smaller_pose],
                            w_condition=[width//8//apply_smaller_pose],
                            # pose
                            h_pose=[height//8//apply_smaller_pose],
                            w_pose=[width//8//apply_smaller_pose],
                            # real
                            h_real=[height//8],
                            w_real=[width//8],
                            device=self.device,
                            prompt=round_prompt, # detailed_prompt
                            clip_prompt=general_prompt, # general_prompt
                            pose_id_value=1,
                            add_pose_clip=True,
                            bounding_boxes_in_order=bounding_boxes_in_order,
                            customize_img_ids=customize_img_ids,
                            customize_txt_ids=customize_txt_ids,
                            generate_img_ids_type=generate_img_ids_type,
                            apply_smaller_pose=apply_smaller_pose
                            )
        return inp_cond

    def prepare_multi_person_version2(self, round_prompt, general_prompt, width, height, customize_img_ids, bounding_boxes_in_order, customize_txt_ids, generate_img_ids_type, embed_id=False, pretend_pe=False, pe_learnable=False, apply_smaller_pose=1):
        if embed_id:
            assert False, "not implemented for version2"
        elif pretend_pe:
            assert False, "not implemented for version2"
        elif pe_learnable:
            assert False, "not implemented for version2"

        else:
            print("prepare_dual_train_ar")
            inp_cond = prepare_dual_train_ar_version2(self.t5,
                            self.clip,
                            bs=1,
                            # condition
                            h_condition=[height//8//apply_smaller_pose],
                            w_condition=[width//8//apply_smaller_pose],
                            # pose
                            h_pose=[height//8//apply_smaller_pose],
                            w_pose=[width//8//apply_smaller_pose],
                            # real
                            h_real=[height//8],
                            w_real=[width//8],
                            # pre_real
                            h_pre_real=[height//8],
                            w_pre_real=[width//8],
                            device=self.device,
                            prompt=round_prompt, # detailed_prompt
                            clip_prompt=general_prompt, # general_prompt
                            pose_id_value=1,
                            add_pose_clip=True,
                            bounding_boxes_in_order=bounding_boxes_in_order,
                            customize_img_ids=customize_img_ids,
                            customize_txt_ids=customize_txt_ids,
                            generate_img_ids_type=generate_img_ids_type,
                            apply_smaller_pose=apply_smaller_pose
                            )
        return inp_cond

    def decode_latent(self, latent, height, width):
        print("Decode latent")
        x = unpack(latent.float(), height, width)
        x = self.ae.decode(x)
        x1 = x.clamp(-1, 1)
        x1 = rearrange(x1[-1], "c h w -> h w c")
        output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
        return output_img

    def decode_round_results(self, round_latent_results, height, width, apply_smaller_pose):
        pose_latent = round_latent_results["pose_latent"]
        img_latent = round_latent_results["img_latent"]
        pose_img = self.decode_latent(pose_latent, height//apply_smaller_pose, width//apply_smaller_pose)
        real_img = self.decode_latent(img_latent, height, width)
        return {"pose_img": pose_img, "real_img": real_img}

    def decode_t2i(self, round_latent_results, height, width):
        img_latent = round_latent_results["img_latent"]
        real_img = self.decode_latent(img_latent, height, width)
        return {"real_img": real_img}



    def create_attention_mask(self, batch_size: int, text_seqlen: int, cond_seqlen: int, pose_seqlen: int, real_seqlen: int) -> torch.Tensor:
        """
        Generates an attention mask for a concatenated sequence.

        The sequence is structured as:
        [text_sequence, condition_sequence, pose_sequence, real_sequence]

        The masking rule applied is:
        - The 'real_sequence' cannot attend to (cannot 'see') the 'condition_sequence'.
        - All other parts of the sequence can fully attend to each other.

        Args:
            text_seqlen (int): The length of the text sequence.
            cond_seqlen (int): The length of the condition sequence.
            pose_seqlen (int): The length of the pose sequence.
            real_seqlen (int): The length of the real sequence.

        Returns:
            torch.Tensor: A 2D PyTorch tensor representing the attention mask.
                        The shape is [total_sequence_length, total_sequence_length].
                        Values are 0.0 for allowed attention and -1e9 for masked attention.
                        This mask can be directly added to attention logits before softmax.
        """
        # Calculate the total length of the combined sequence
        total_len = text_seqlen + cond_seqlen + pose_seqlen + real_seqlen

        # Initialize the mask with zeros using PyTorch.
        # Zeros indicate full attention (no masking).
        # We use float32 because these masks are typically added to attention scores.
        mask = torch.zeros((batch_size, total_len, total_len), dtype=torch.float32)

        # Define the start and end indices for each segment within the total sequence.
        # These indices are crucial for correctly slicing the mask.
        # text_sequence occupies indices [0, text_seqlen - 1]
        # condition_sequence occupies indices [text_seqlen, text_seqlen + cond_seqlen - 1]
        # pose_sequence occupies indices [text_seqlen + cond_seqlen, text_seqlen + cond_seqlen + pose_seqlen - 1]
        # real_sequence occupies indices [text_seqlen + cond_seqlen + pose_seqlen, total_len - 1]

        # Calculate the start and end indices for the 'condition_sequence' keys
        # (the part that 'real_sequence' queries should not see).
        cond_keys_start = text_seqlen
        cond_keys_end = text_seqlen + cond_seqlen

        # Calculate the start and end indices for the 'real_sequence' queries
        # (the part that should not see 'condition_sequence' keys).
        real_queries_start = text_seqlen + cond_seqlen + pose_seqlen
        real_queries_end = total_len

        # Apply the masking rule: real_sequence queries cannot see condition_sequence keys.
        # In an attention mask M[i, j], 'i' is the query index and 'j' is the key index.
        # Setting a value to -1e9 (a very small number) effectively makes its
        # softmax probability zero after exponentiation, thus masking it out.
        # PyTorch tensors support direct slicing and assignment, no need for numpy conversion.
        mask[:, real_queries_start:real_queries_end, cond_keys_start:cond_keys_end] = -1e9

        return mask

    def create_attention_mask_version2(self, batch_size: int, text_seqlen: int, cond_seqlen: int, pose_seqlen: int, real_seqlen: int) -> torch.Tensor:
        """
        Generates an attention mask for a concatenated sequence.

        The sequence is structured as:
        [text_sequence, condition_sequence, pose_sequence, real_sequence]

        The masking rule applied is:
        - The 'real_sequence' cannot attend to (cannot 'see') the 'condition_sequence'.
        - All other parts of the sequence can fully attend to each other.

        Args:
            text_seqlen (int): The length of the text sequence.
            cond_seqlen (int): The length of the condition sequence.
            pose_seqlen (int): The length of the pose sequence.
            real_seqlen (int): The length of the real sequence.

        Returns:
            torch.Tensor: A 2D PyTorch tensor representing the attention mask.
                        The shape is [total_sequence_length, total_sequence_length].
                        Values are 0.0 for allowed attention and -1e9 for masked attention.
                        This mask can be directly added to attention logits before softmax.
        """
        # Calculate the total length of the combined sequence
        total_len = text_seqlen + cond_seqlen + pose_seqlen + real_seqlen + real_seqlen

        # Initialize the mask with zeros using PyTorch.
        # Zeros indicate full attention (no masking).
        # We use float32 because these masks are typically added to attention scores.
        mask = torch.zeros((batch_size, total_len, total_len), dtype=torch.float32)

        # Define the start and end indices for each segment within the total sequence.
        # These indices are crucial for correctly slicing the mask.
        # text_sequence occupies indices [0, text_seqlen - 1]
        # condition_sequence occupies indices [text_seqlen, text_seqlen + cond_seqlen - 1]
        # pose_sequence occupies indices [text_seqlen + cond_seqlen, text_seqlen + cond_seqlen + pose_seqlen - 1]
        # real_sequence occupies indices [text_seqlen + cond_seqlen + pose_seqlen, total_len - 1]

        # Calculate the start and end indices for the 'condition_sequence' keys
        # (the part that 'real_sequence' queries should not see).
        cond_keys_start = text_seqlen
        cond_keys_end = text_seqlen + cond_seqlen

        # Calculate the start and end indices for the 'real_sequence' queries
        # (the part that should not see 'condition_sequence' keys).
        real_queries_start = text_seqlen + cond_seqlen + pose_seqlen + real_seqlen
        real_queries_end = total_len

        # Apply the masking rule: real_sequence queries cannot see condition_sequence keys.
        # In an attention mask M[i, j], 'i' is the query index and 'j' is the key index.
        # Setting a value to -1e9 (a very small number) effectively makes its
        # softmax probability zero after exponentiation, thus masking it out.
        # PyTorch tensors support direct slicing and assignment, no need for numpy conversion.
        mask[:, real_queries_start:real_queries_end, cond_keys_start:cond_keys_end] = -1e9

        return mask

    def forward_multiperson_morerounds(self,
                            prompts: list[str],
                            general_prompt: str,
                            width: int = 512,
                            height: int = 512,
                            guidance: float = 4,
                            num_steps: int = 50,
                            seed: int = 123456789,
                            customize_img_ids: bool = False,
                            customize_txt_ids: bool = False,
                            bounding_boxes_in_order: list[list[int]] = None,
                            generate_img_ids_type: int = 1,
                            background_color: str = "black",
                            use_v1_bbox: bool = True, # if true, use previous v1-inference [bbox]
                            attn_mask_type: str = None, # None default -- if None, draw back to json2.py
                            embed_id: bool = False,
                            num_people_indicator: bool = False,
                            pretend_pe: bool = False,
                            pe_learnable: bool = False,
                            learned_shift_scale_ratio: float = 1,
                            ori_h: Tensor | None = None,
                            ori_w: Tensor | None = None,
                            apply_smaller_pose: int = 1,
                            ):
        round_results = {}
        print("width!!!!", width)
        print("height!!!", height)
        noise_real = self.get_noise_patchified(height, width, seed) # [1, 1024, 64]
        noise_pose = torch.clone(noise_real)
        black_latents = self.get_condition_latents_patchified(width//apply_smaller_pose, height//apply_smaller_pose, background_color) # [1, 1024, 64]


        ##### deal for attn_mask_type #### do nothing but just full attention if attn_mask_type is None
        text_seqlen = 512
        cond_seqlen = black_latents.shape[1]
        pose_seqlen = noise_pose.shape[1]
        real_seqlen = noise_real.shape[1]
        batch_size = noise_pose.shape[0]
        if attn_mask_type is None:
            ar_attn_mask = None
        else:
            if attn_mask_type == "mask_condition_for_real":
                ar_attn_mask = self.create_attention_mask(batch_size=batch_size, text_seqlen=text_seqlen, cond_seqlen=cond_seqlen, pose_seqlen=pose_seqlen, real_seqlen=real_seqlen)
                ar_attn_mask = ar_attn_mask.to(noise_pose.device)
                ar_attn_mask = ar_attn_mask.to(noise_pose.dtype)
            else:
                raise ValueError(f"Invalid attn_mask_type: {attn_mask_type}")

        #############################################################################################

        timesteps = get_schedule(
            num_steps,
            # (width // 8) * (height // 8) // (16 * 16),
            (width // 16) * (height // 16),
            shift=True,
        )
        torch.manual_seed(seed)

        # first round
        print("first round")
        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            input_prompt = [prompts[0]] if not customize_txt_ids else [[prompts[0]]]
            input_bounding_boxes = [[bounding_boxes_in_order[0]]] if bounding_boxes_in_order is not None else None
            inp_cond = self.prepare_multi_person(input_prompt, general_prompt, width, height, customize_img_ids, input_bounding_boxes, customize_txt_ids, generate_img_ids_type, embed_id, False, False, apply_smaller_pose)
            x_t = torch.cat([black_latents, noise_pose, noise_real], dim=1)
            visual_ids = torch.cat([inp_cond['img_ids_condition'], inp_cond['img_ids_pose'], inp_cond['img_ids_real']], dim=1)
            bbox_ids = None
            guidance_vec = torch.full((x_t.shape[0],), guidance, device=x_t.device, dtype=x_t.dtype)

            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)

            real_seqlen = noise_real.shape[1]
            print("black_latents.shape", black_latents.shape)
            print("noise_pose.shape", noise_pose.shape)
            print("noise_real.shape", noise_real.shape)
            assert black_latents.shape[1] == noise_pose.shape[1]
            pose_seqlen = black_latents.shape[1]
            output1 = self.denoise_oneround(
                img=x_t,
                img_ids=visual_ids,
                bbox_ids=bbox_ids,
                txt=inp_cond['txt'],
                txt_ids=inp_cond['txt_ids'],
                vec=inp_cond['vec'],
                vec_pose=inp_cond['vec_pose'],
                timesteps=timesteps,
                guidance_vec=guidance_vec,
                image_sequence_len=real_seqlen,
                cond_sequence_len=pose_seqlen,
                ar_attn_mask=ar_attn_mask,
                embed_id=embed_id,
                pe_learnable=pe_learnable,
                learned_shift_scale_ratio=learned_shift_scale_ratio,
                batch_multi_person_last_indicator=torch.tensor([False], device=self.device),
            )
            #print("output1.shape", output1.shape)
            output1_pose_latent = output1[:, pose_seqlen:2*pose_seqlen, :]
            #print("output1_pose_latent.shape", output1_pose_latent.shape)
            output1_img_latent = output1[:, 2*pose_seqlen:, :]
            #print("output1_pose_latent.shape", output1_pose_latent.shape)
            #print("shape of [output1_pose_latent, output1_img_latent]", output1_pose_latent.shape, output1_img_latent.shape)
            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(output1_pose_latent.device)
            round_results["1"] = self.decode_round_results({"pose_latent": output1_pose_latent, "img_latent": output1_img_latent}, height, width, apply_smaller_pose)


            # second round to the last round
            len_people = len(prompts)
            for round in range(1, len_people):
                print(f"round {round + 1}")
                input_prompt = [prompts[:round+1]]
                if bounding_boxes_in_order is not None:
                    input_bounding_boxes = [bounding_boxes_in_order[:round+1]]
                else:
                    input_bounding_boxes = None
                x_t = torch.cat([output1_pose_latent, noise_pose, noise_real], dim=1)
                if self.offload:
                    self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
                inp_cond = self.prepare_multi_person(input_prompt, general_prompt, width, height, customize_img_ids, input_bounding_boxes, customize_txt_ids, generate_img_ids_type, embed_id, pretend_pe, pe_learnable, apply_smaller_pose)
                visual_ids = torch.cat([inp_cond['img_ids_condition'], inp_cond['img_ids_pose'], inp_cond['img_ids_real']], dim=1)
                bbox_ids = None
                if self.offload:
                    self.offload_model_to_cpu(self.t5, self.clip)
                    self.model = self.model.to(self.device)
                output2 = self.denoise_oneround(
                    img=x_t,
                    img_ids=visual_ids,
                    bbox_ids=bbox_ids,
                    txt=inp_cond['txt'],
                    txt_ids=inp_cond['txt_ids'],
                    vec=inp_cond['vec'],
                    vec_pose=inp_cond['vec_pose'],
                    timesteps=timesteps,
                    guidance_vec=guidance_vec,
                    image_sequence_len=real_seqlen,
                    cond_sequence_len=pose_seqlen,
                    ar_attn_mask=ar_attn_mask,
                    embed_id=embed_id,
                    pe_learnable=pe_learnable,
                    learned_shift_scale_ratio=learned_shift_scale_ratio,
                    batch_multi_person_last_indicator=torch.tensor([True], device=self.device),
                    ori_h=ori_h,
                    ori_w=ori_w,
                )
                output2_pose_latent = output2[:, pose_seqlen:2*pose_seqlen, :]
                output2_img_latent = output2[:, 2*pose_seqlen:, :]
                print("shape of [output2_pose_latent, output2_img_latent]", output2_pose_latent.shape, output2_img_latent.shape)
                if self.offload:
                    self.offload_model_to_cpu(self.model)
                    self.ae.decoder.to(output1_pose_latent.device)
                round_results[f"{round + 1}"] = self.decode_round_results({"pose_latent": output2_pose_latent, "img_latent": output2_img_latent}, height, width, apply_smaller_pose)
                output1_pose_latent = output2_pose_latent
            return round_results, self.clip, self.t5, self.ae

    def forward_multiperson_morerounds_rewrite(self,
                            prompts: list[str],
                            rewrite_prompts: list[dict],
                            general_prompt: str,
                            width: int = 512,
                            height: int = 512,
                            guidance: float = 4,
                            num_steps: int = 50,
                            seed: int = 123456789,
                            customize_img_ids: bool = False,
                            customize_txt_ids: bool = False,
                            bounding_boxes_in_order: list[list[int]] = None,
                            generate_img_ids_type: int = 1,
                            background_color: str = "black",
                            use_v1_bbox: bool = True, # if true, use previous v1-inference [bbox]
                            attn_mask_type: str = None, # None default -- if None, draw back to json2.py
                            embed_id: bool = False,
                            num_people_indicator: bool = False,
                            pretend_pe: bool = False,
                            pe_learnable: bool = False,
                            learned_shift_scale_ratio: float = 1,
                            ori_h: Tensor | None = None,
                            ori_w: Tensor | None = None,
                            apply_smaller_pose: int = 1,
                            ):
        round_results = {}
        print("width!!!!", width)
        print("height!!!", height)
        noise_real = self.get_noise_patchified(height, width, seed) # [1, 1024, 64]
        noise_pose = torch.clone(noise_real)
        black_latents = self.get_condition_latents_patchified(width//apply_smaller_pose, height//apply_smaller_pose, background_color) # [1, 1024, 64]


        ##### deal for attn_mask_type #### do nothing but just full attention if attn_mask_type is None
        text_seqlen = 512
        cond_seqlen = black_latents.shape[1]
        pose_seqlen = noise_pose.shape[1]
        real_seqlen = noise_real.shape[1]
        batch_size = noise_pose.shape[0]
        if attn_mask_type is None:
            ar_attn_mask = None
        else:
            if attn_mask_type == "mask_condition_for_real":
                ar_attn_mask = self.create_attention_mask(batch_size=batch_size, text_seqlen=text_seqlen, cond_seqlen=cond_seqlen, pose_seqlen=pose_seqlen, real_seqlen=real_seqlen)
                ar_attn_mask = ar_attn_mask.to(noise_pose.device)
                ar_attn_mask = ar_attn_mask.to(noise_pose.dtype)
            else:
                raise ValueError(f"Invalid attn_mask_type: {attn_mask_type}")

        #############################################################################################

        timesteps = get_schedule(
            num_steps,
            # (width // 8) * (height // 8) // (16 * 16),
            (width // 16) * (height // 16),
            shift=True,
        )
        torch.manual_seed(seed)

        # first round
        print("first round")
        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            input_prompt = [prompts[0]] if not customize_txt_ids else [[prompts[0]]]
            # ------
            rewrite_full_prompt = [rewrite_prompts[0]['caption']]
            rewrite_tag = [[item['tag'] for item in rewrite_prompts[0]["segments"]]]
            rewrite_segment = [[item["text"] for item in rewrite_prompts[0]["segments"]]]
            # ------
            input_bounding_boxes = [[bounding_boxes_in_order[0]]] if bounding_boxes_in_order is not None else None
            inp_cond = self.prepare_multi_person_rewrite(input_prompt, rewrite_full_prompt, rewrite_tag, rewrite_segment, general_prompt, width, height, customize_img_ids, input_bounding_boxes, customize_txt_ids, generate_img_ids_type, embed_id, False, False, apply_smaller_pose)
            x_t = torch.cat([black_latents, noise_pose, noise_real], dim=1)
            visual_ids = torch.cat([inp_cond['img_ids_condition'], inp_cond['img_ids_pose'], inp_cond['img_ids_real']], dim=1)
            bbox_ids = None
            guidance_vec = torch.full((x_t.shape[0],), guidance, device=x_t.device, dtype=x_t.dtype)

            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)

            real_seqlen = noise_real.shape[1]
            print("black_latents.shape", black_latents.shape)
            print("noise_pose.shape", noise_pose.shape)
            print("noise_real.shape", noise_real.shape)
            assert black_latents.shape[1] == noise_pose.shape[1]
            pose_seqlen = black_latents.shape[1]
            output1 = self.denoise_oneround(
                img=x_t,
                img_ids=visual_ids,
                bbox_ids=bbox_ids,
                txt=inp_cond['txt'],
                txt_ids=inp_cond['txt_ids'],
                vec=inp_cond['vec'],
                vec_pose=inp_cond['vec_pose'],
                timesteps=timesteps,
                guidance_vec=guidance_vec,
                image_sequence_len=real_seqlen,
                cond_sequence_len=pose_seqlen,
                ar_attn_mask=ar_attn_mask,
                embed_id=embed_id,
                pe_learnable=pe_learnable,
                learned_shift_scale_ratio=learned_shift_scale_ratio,
                batch_multi_person_last_indicator=torch.tensor([False], device=self.device),
            )
            #print("output1.shape", output1.shape)
            output1_pose_latent = output1[:, pose_seqlen:2*pose_seqlen, :]
            #print("output1_pose_latent.shape", output1_pose_latent.shape)
            output1_img_latent = output1[:, 2*pose_seqlen:, :]
            #print("output1_pose_latent.shape", output1_pose_latent.shape)
            #print("shape of [output1_pose_latent, output1_img_latent]", output1_pose_latent.shape, output1_img_latent.shape)
            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(output1_pose_latent.device)
            round_results["1"] = self.decode_round_results({"pose_latent": output1_pose_latent, "img_latent": output1_img_latent}, height, width, apply_smaller_pose)


            # second round to the last round
            len_people = len(prompts)
            for round in range(1, len_people):
                print(f"round {round + 1}")
                input_prompt = [prompts[:round+1]]
                # ------
                rewrite_full_prompt = [rewrite_prompts[round]['caption']]
                rewrite_tag = [[item['tag'] for item in rewrite_prompts[round]["segments"]]]
                rewrite_segment = [[item["text"] for item in rewrite_prompts[round]["segments"]]]
                # ------
                if bounding_boxes_in_order is not None:
                    input_bounding_boxes = [bounding_boxes_in_order[:round+1]]
                else:
                    input_bounding_boxes = None
                x_t = torch.cat([output1_pose_latent, noise_pose, noise_real], dim=1)
                if self.offload:
                    self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
                inp_cond = self.prepare_multi_person_rewrite(input_prompt, rewrite_full_prompt, rewrite_tag, rewrite_segment, general_prompt, width, height, customize_img_ids, input_bounding_boxes, customize_txt_ids, generate_img_ids_type, embed_id, pretend_pe, pe_learnable, apply_smaller_pose)
                visual_ids = torch.cat([inp_cond['img_ids_condition'], inp_cond['img_ids_pose'], inp_cond['img_ids_real']], dim=1)
                bbox_ids = None
                if self.offload:
                    self.offload_model_to_cpu(self.t5, self.clip)
                    self.model = self.model.to(self.device)
                output2 = self.denoise_oneround(
                    img=x_t,
                    img_ids=visual_ids,
                    bbox_ids=bbox_ids,
                    txt=inp_cond['txt'],
                    txt_ids=inp_cond['txt_ids'],
                    vec=inp_cond['vec'],
                    vec_pose=inp_cond['vec_pose'],
                    timesteps=timesteps,
                    guidance_vec=guidance_vec,
                    image_sequence_len=real_seqlen,
                    cond_sequence_len=pose_seqlen,
                    ar_attn_mask=ar_attn_mask,
                    embed_id=embed_id,
                    pe_learnable=pe_learnable,
                    learned_shift_scale_ratio=learned_shift_scale_ratio,
                    batch_multi_person_last_indicator=torch.tensor([True], device=self.device),
                    ori_h=ori_h,
                    ori_w=ori_w,
                )
                output2_pose_latent = output2[:, pose_seqlen:2*pose_seqlen, :]
                output2_img_latent = output2[:, 2*pose_seqlen:, :]
                print("shape of [output2_pose_latent, output2_img_latent]", output2_pose_latent.shape, output2_img_latent.shape)
                if self.offload:
                    self.offload_model_to_cpu(self.model)
                    self.ae.decoder.to(output1_pose_latent.device)
                round_results[f"{round + 1}"] = self.decode_round_results({"pose_latent": output2_pose_latent, "img_latent": output2_img_latent}, height, width, apply_smaller_pose)
                output1_pose_latent = output2_pose_latent
            return round_results, self.clip, self.t5, self.ae



    def forward_multiperson_oneshot(self,
                            prompts: list[str],
                            general_prompt: str,
                            width: int = 512,
                            height: int = 512,
                            guidance: float = 4,
                            num_steps: int = 50,
                            seed: int = 123456789,
                            customize_img_ids: bool = False,
                            customize_txt_ids: bool = False,
                            bounding_boxes_in_order: list[list[int]] = None,
                            generate_img_ids_type: int = 1,
                            background_color: str = "black",
                            use_v1_bbox: bool = True, # if true, use previous v1-inference [bbox]
                            attn_mask_type: str = None, # None default -- if None, draw back to json2.py
                            embed_id: bool = False,
                            num_people_indicator: bool = False,
                            pretend_pe: bool = False,
                            pe_learnable: bool = False,
                            learned_shift_scale_ratio: float = 1,
                            ori_h: Tensor | None = None,
                            ori_w: Tensor | None = None,
                            apply_smaller_pose: int = 1,
                            ):
        round_results = {}
        print("width!!!!", width)
        print("height!!!", height)
        noise_real = self.get_noise_patchified(height, width, seed) # [1, 1024, 64]
        noise_pose = torch.clone(noise_real)


        ##### deal for attn_mask_type #### do nothing but just full attention if attn_mask_type is None
        text_seqlen = 512
        pose_seqlen = noise_pose.shape[1]
        real_seqlen = noise_real.shape[1]
        batch_size = noise_pose.shape[0]
        ar_attn_mask = None
        #############################################################################################

        timesteps = get_schedule(
            num_steps,
            # (width // 8) * (height // 8) // (16 * 16),
            (width // 16) * (height // 16),
            shift=True,
        )
        torch.manual_seed(seed)

        # first round
        print("one shot")
        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            input_prompt = [" ".join(prompts)] if not customize_txt_ids else [prompts]
            input_bounding_boxes = [bounding_boxes_in_order] if bounding_boxes_in_order is not None else None
            inp_cond = self.prepare_multi_person(input_prompt, general_prompt, width, height, customize_img_ids, input_bounding_boxes, customize_txt_ids, generate_img_ids_type, embed_id, False, False, apply_smaller_pose, oneshot=True)
            x_t = torch.cat([noise_pose, noise_real], dim=1)
            visual_ids = torch.cat([inp_cond['img_ids_pose'], inp_cond['img_ids_real']], dim=1)
            bbox_ids = None
            guidance_vec = torch.full((x_t.shape[0],), guidance, device=x_t.device, dtype=x_t.dtype)

            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)

            real_seqlen = noise_real.shape[1]
            print("noise_pose.shape", noise_pose.shape)
            print("noise_real.shape", noise_real.shape)
            pose_seqlen = noise_pose.shape[1]
            output1 = self.denoise_oneround_oneshot(
                img=x_t,
                img_ids=visual_ids,
                bbox_ids=bbox_ids,
                txt=inp_cond['txt'],
                txt_ids=inp_cond['txt_ids'],
                vec=inp_cond['vec'],
                vec_pose=inp_cond['vec_pose'],
                timesteps=timesteps,
                guidance_vec=guidance_vec,
                image_sequence_len=real_seqlen,
                batch_multi_person_last_indicator=torch.tensor([False], device=self.device),
            )
            #print("output1.shape", output1.shape)
            output1_pose_latent = output1[:, :pose_seqlen, :]
            #print("output1_pose_latent.shape", output1_pose_latent.shape)
            output1_img_latent = output1[:, pose_seqlen:, :]
            #print("output1_pose_latent.shape", output1_pose_latent.shape)
            #print("shape of [output1_pose_latent, output1_img_latent]", output1_pose_latent.shape, output1_img_latent.shape)
            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(output1_pose_latent.device)
            round_results["1"] = self.decode_round_results({"pose_latent": output1_pose_latent, "img_latent": output1_img_latent}, height, width, apply_smaller_pose)
            return round_results, self.clip, self.t5, self.ae

    def forward_multiperson_oneshot_allprompt(self,
                            prompts: list[str],
                            general_prompt: str,
                            width: int = 512,
                            height: int = 512,
                            guidance: float = 4,
                            num_steps: int = 50,
                            seed: int = 123456789,
                            customize_img_ids: bool = False,
                            customize_txt_ids: bool = False,
                            bounding_boxes_in_order: list[list[int]] = None,
                            generate_img_ids_type: int = 1,
                            background_color: str = "black",
                            use_v1_bbox: bool = True, # if true, use previous v1-inference [bbox]
                            attn_mask_type: str = None, # None default -- if None, draw back to json2.py
                            embed_id: bool = False,
                            num_people_indicator: bool = False,
                            pretend_pe: bool = False,
                            pe_learnable: bool = False,
                            learned_shift_scale_ratio: float = 1,
                            ori_h: Tensor | None = None,
                            ori_w: Tensor | None = None,
                            apply_smaller_pose: int = 1,
                            ):
        round_results = {}
        print("width!!!!", width)
        print("height!!!", height)
        noise_real = self.get_noise_patchified(height, width, seed) # [1, 1024, 64]
        noise_pose = torch.clone(noise_real)


        ##### deal for attn_mask_type #### do nothing but just full attention if attn_mask_type is None
        text_seqlen = 512
        pose_seqlen = noise_pose.shape[1]
        real_seqlen = noise_real.shape[1]
        batch_size = noise_pose.shape[0]
        ar_attn_mask = None
        #############################################################################################

        timesteps = get_schedule(
            num_steps,
            # (width // 8) * (height // 8) // (16 * 16),
            (width // 16) * (height // 16),
            shift=True,
        )
        torch.manual_seed(seed)

        # first round
        print("one shot")
        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            input_prompt = [general_prompt + " " + " ".join(prompts)] if not customize_txt_ids else [prompts]
            input_bounding_boxes = [bounding_boxes_in_order] if bounding_boxes_in_order is not None else None
            inp_cond = self.prepare_multi_person(input_prompt, general_prompt, width, height, customize_img_ids, input_bounding_boxes, customize_txt_ids, generate_img_ids_type, embed_id, False, False, apply_smaller_pose, oneshot=True)
            x_t = torch.cat([noise_pose, noise_real], dim=1)
            visual_ids = torch.cat([inp_cond['img_ids_pose'], inp_cond['img_ids_real']], dim=1)
            bbox_ids = None
            guidance_vec = torch.full((x_t.shape[0],), guidance, device=x_t.device, dtype=x_t.dtype)

            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)

            real_seqlen = noise_real.shape[1]
            print("noise_pose.shape", noise_pose.shape)
            print("noise_real.shape", noise_real.shape)
            pose_seqlen = noise_pose.shape[1]
            output1 = self.denoise_oneround_oneshot(
                img=x_t,
                img_ids=visual_ids,
                bbox_ids=bbox_ids,
                txt=inp_cond['txt'],
                txt_ids=inp_cond['txt_ids'],
                vec=inp_cond['vec'],
                vec_pose=inp_cond['vec_pose'],
                timesteps=timesteps,
                guidance_vec=guidance_vec,
                image_sequence_len=real_seqlen,
                batch_multi_person_last_indicator=torch.tensor([False], device=self.device),
            )
            #print("output1.shape", output1.shape)
            output1_pose_latent = output1[:, :pose_seqlen, :]
            #print("output1_pose_latent.shape", output1_pose_latent.shape)
            output1_img_latent = output1[:, pose_seqlen:, :]
            #print("output1_pose_latent.shape", output1_pose_latent.shape)
            #print("shape of [output1_pose_latent, output1_img_latent]", output1_pose_latent.shape, output1_img_latent.shape)
            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(output1_pose_latent.device)
            round_results["1"] = self.decode_round_results({"pose_latent": output1_pose_latent, "img_latent": output1_img_latent}, height, width, apply_smaller_pose)
            return round_results, self.clip, self.t5, self.ae


    def forward_multiperson_t2i(self,
                            prompts: list[str],
                            general_prompt: str,
                            width: int = 512,
                            height: int = 512,
                            guidance: float = 4,
                            num_steps: int = 50,
                            seed: int = 123456789,
                            customize_img_ids: bool = False,
                            customize_txt_ids: bool = False,
                            bounding_boxes_in_order: list[list[int]] = None,
                            generate_img_ids_type: int = 1,
                            background_color: str = "black",
                            use_v1_bbox: bool = True, # if true, use previous v1-inference [bbox]
                            attn_mask_type: str = None, # None default -- if None, draw back to json2.py
                            embed_id: bool = False,
                            num_people_indicator: bool = False,
                            pretend_pe: bool = False,
                            pe_learnable: bool = False,
                            learned_shift_scale_ratio: float = 1,
                            ori_h: Tensor | None = None,
                            ori_w: Tensor | None = None,
                            apply_smaller_pose: int = 1,
                            ):
        round_results = {}
        print("width!!!!", width)
        print("height!!!", height)
        noise_real = self.get_noise_patchified(height, width, seed)


        ##### deal for attn_mask_type #### do nothing but just full attention if attn_mask_type is None
        text_seqlen = 512
        real_seqlen = noise_real.shape[1]
        batch_size = noise_real.shape[0]
        ar_attn_mask = None
        #############################################################################################

        timesteps = get_schedule(
            num_steps,
            # (width // 8) * (height // 8) // (16 * 16),
            (width // 16) * (height // 16),
            shift=True,
        )
        torch.manual_seed(seed)

        # first round
        print("t2i")
        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            input_prompt = [" ".join(prompts)] if not customize_txt_ids else [prompts]
            input_bounding_boxes = [bounding_boxes_in_order] if bounding_boxes_in_order is not None else None
            inp_cond = self.prepare_multi_person(input_prompt, general_prompt, width, height, customize_img_ids, input_bounding_boxes, customize_txt_ids, generate_img_ids_type, embed_id, False, False, apply_smaller_pose, t2i=True)
            x_t = noise_real
            visual_ids = inp_cond['img_ids_real']
            bbox_ids = None
            guidance_vec = torch.full((x_t.shape[0],), guidance, device=x_t.device, dtype=x_t.dtype)

            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)

            real_seqlen = noise_real.shape[1]
            print("noise_real.shape", noise_real.shape)
            output1 = self.denoise_t2i(
                img=x_t,
                img_ids=visual_ids,
                bbox_ids=bbox_ids,
                txt=inp_cond['txt'],
                txt_ids=inp_cond['txt_ids'],
                vec=inp_cond['vec'],
                timesteps=timesteps,
                guidance_vec=guidance_vec,
                image_sequence_len=real_seqlen,
                batch_multi_person_last_indicator=torch.tensor([False], device=self.device),
            )
            #print("output1.shape", output1.shape)
            #print("output1_pose_latent.shape", output1_pose_latent.shape)
            output1_img_latent = output1
            #print("output1_pose_latent.shape", output1_pose_latent.shape)
            #print("shape of [output1_pose_latent, output1_img_latent]", output1_pose_latent.shape, output1_img_latent.shape)
            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(output1_img_latent.device)
            round_results["1"] = self.decode_t2i({"img_latent": output1_img_latent}, height, width)
            return round_results, self.clip, self.t5, self.ae

    def forward_multiperson_morerounds_fighting(self,
                            prompts: list[str],
                            general_prompt: str,
                            width: int = 512,
                            height: int = 512,
                            guidance: float = 4,
                            num_steps: int = 50,
                            seed: int = 123456789,
                            customize_img_ids: bool = False,
                            customize_txt_ids: bool = False,
                            bounding_boxes_in_order: list[list[int]] = None,
                            generate_img_ids_type: int = 1,
                            background_color: str = "black",
                            use_v1_bbox: bool = True, # if true, use previous v1-inference [bbox]
                            attn_mask_type: str = None, # None default -- if None, draw back to json2.py
                            embed_id: bool = False,
                            num_people_indicator: bool = False,
                            pretend_pe: bool = False,
                            pe_learnable: bool = False,
                            learned_shift_scale_ratio: float = 1,
                            ori_h: Tensor | None = None,
                            ori_w: Tensor | None = None,
                            apply_smaller_pose: int = 1,
                            ):
        round_results = {}
        print("width!!!!", width)
        print("height!!!", height)
        noise_real = self.get_noise_patchified(height, width, seed) # [1, 1024, 64]
        noise_pose = torch.clone(noise_real)
        black_latents = self.get_condition_latents_patchified(width//apply_smaller_pose, height//apply_smaller_pose, background_color) # [1, 1024, 64]


        ##### deal for attn_mask_type #### do nothing but just full attention if attn_mask_type is None
        text_seqlen = 512
        cond_seqlen = black_latents.shape[1]
        pose_seqlen = noise_pose.shape[1]
        real_seqlen = noise_real.shape[1]
        batch_size = noise_pose.shape[0]
        if attn_mask_type is None:
            ar_attn_mask = None
        else:
            if attn_mask_type == "mask_condition_for_real":
                ar_attn_mask = self.create_attention_mask(batch_size=batch_size, text_seqlen=text_seqlen, cond_seqlen=cond_seqlen, pose_seqlen=pose_seqlen, real_seqlen=real_seqlen)
                ar_attn_mask = ar_attn_mask.to(noise_pose.device)
                ar_attn_mask = ar_attn_mask.to(noise_pose.dtype)
            else:
                raise ValueError(f"Invalid attn_mask_type: {attn_mask_type}")

        #############################################################################################

        timesteps = get_schedule(
            num_steps,
            # (width // 8) * (height // 8) // (16 * 16),
            (width // 16) * (height // 16),
            shift=True,
        )
        torch.manual_seed(seed)

        # first round
        print("first round")
        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            input_prompt = [prompts[0]] if not customize_txt_ids else [[prompts[0]]]
            input_bounding_boxes = [[bounding_boxes_in_order[0]]] if bounding_boxes_in_order is not None else None
            inp_cond = self.prepare_multi_person(input_prompt, general_prompt, width, height, customize_img_ids, input_bounding_boxes, customize_txt_ids, generate_img_ids_type, embed_id, False, False, apply_smaller_pose)
            x_t = torch.cat([black_latents, noise_pose, noise_real], dim=1)
            visual_ids = torch.cat([inp_cond['img_ids_condition'], inp_cond['img_ids_pose'], inp_cond['img_ids_real']], dim=1)
            bbox_ids = None
            guidance_vec = torch.full((x_t.shape[0],), guidance, device=x_t.device, dtype=x_t.dtype)

            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)

            real_seqlen = noise_real.shape[1]
            print("black_latents.shape", black_latents.shape)
            print("noise_pose.shape", noise_pose.shape)
            print("noise_real.shape", noise_real.shape)
            assert black_latents.shape[1] == noise_pose.shape[1]
            pose_seqlen = black_latents.shape[1]

            output1 = self.denoise_oneround(
                img=x_t,
                img_ids=visual_ids,
                bbox_ids=bbox_ids,
                txt=inp_cond['txt'],
                txt_ids=inp_cond['txt_ids'],
                vec=inp_cond['vec'],
                vec_pose=inp_cond['vec_pose'],
                timesteps=timesteps,
                guidance_vec=guidance_vec,
                image_sequence_len=real_seqlen,
                cond_sequence_len=pose_seqlen,
                ar_attn_mask=ar_attn_mask,
                embed_id=embed_id,
                pe_learnable=pe_learnable,
                learned_shift_scale_ratio=learned_shift_scale_ratio,
                batch_multi_person_last_indicator=torch.tensor([False], device=self.device),
            )
            #print("output1.shape", output1.shape)
            output1_pose_latent = output1[:, pose_seqlen:2*pose_seqlen, :]
            #print("output1_pose_latent.shape", output1_pose_latent.shape)
            output1_img_latent = output1[:, 2*pose_seqlen:, :]
            #print("output1_pose_latent.shape", output1_pose_latent.shape)
            #print("shape of [output1_pose_latent, output1_img_latent]", output1_pose_latent.shape, output1_img_latent.shape)
            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(output1_pose_latent.device)
            round_results["1"] = self.decode_round_results({"pose_latent": output1_pose_latent, "img_latent": output1_img_latent}, height, width, apply_smaller_pose)

            # second round to the last round
            len_people = len(prompts)
            for round in range(1, len_people):
                print(f"round {round + 1}")
                input_prompt = [prompts[:round+1]]
                if bounding_boxes_in_order is not None:
                    input_bounding_boxes = [bounding_boxes_in_order[:round+1]]
                else:
                    input_bounding_boxes = None

                noise_real_w_partial_init, t_map = init_partial_noise_with_boxes_patchified(
                    noise_tokens=noise_real,
                    x_1=output1_img_latent,
                    input_bounding_boxes=input_bounding_boxes,
                    img_size=(height, width),
                    already_denoised_frac=0.30,  # "already denoised 30%" inside boxes[:-1]
                    blur_sigma=1.0,
                    exclude_last=True,
                    seed=7,
                )
                x_t = torch.cat([output1_pose_latent, noise_pose, noise_real_w_partial_init], dim=1)
                if self.offload:
                    self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
                inp_cond = self.prepare_multi_person(input_prompt, general_prompt, width, height, customize_img_ids, input_bounding_boxes, customize_txt_ids, generate_img_ids_type, embed_id, pretend_pe, pe_learnable, apply_smaller_pose)
                visual_ids = torch.cat([inp_cond['img_ids_condition'], inp_cond['img_ids_pose'], inp_cond['img_ids_real']], dim=1)
                bbox_ids = None
                if self.offload:
                    self.offload_model_to_cpu(self.t5, self.clip)
                    self.model = self.model.to(self.device)

                # timesteps = get_schedule(
                #     100,
                #     # (width // 8) * (height // 8) // (16 * 16),
                #     (width // 16) * (height // 16),
                #     shift=True,
                # )
                # guidance_vec = torch.full((x_t.shape[0],), 4, device=x_t.device, dtype=x_t.dtype)
                output2 = self.denoise_oneround(
                    img=x_t,
                    img_ids=visual_ids,
                    bbox_ids=bbox_ids,
                    txt=inp_cond['txt'],
                    txt_ids=inp_cond['txt_ids'],
                    vec=inp_cond['vec'],
                    vec_pose=inp_cond['vec_pose'],
                    timesteps=timesteps,
                    guidance_vec=guidance_vec,
                    image_sequence_len=real_seqlen,
                    cond_sequence_len=pose_seqlen,
                    ar_attn_mask=ar_attn_mask,
                    embed_id=embed_id,
                    pe_learnable=pe_learnable,
                    learned_shift_scale_ratio=learned_shift_scale_ratio,
                    batch_multi_person_last_indicator=torch.tensor([True], device=self.device),
                    ori_h=ori_h,
                    ori_w=ori_w,
                )
                output2_pose_latent = output2[:, pose_seqlen:2*pose_seqlen, :]
                output2_img_latent = output2[:, 2*pose_seqlen:, :]
                print("shape of [output2_pose_latent, output2_img_latent]", output2_pose_latent.shape, output2_img_latent.shape)
                if self.offload:
                    self.offload_model_to_cpu(self.model)
                    self.ae.decoder.to(output1_pose_latent.device)
                round_results[f"{round + 1}"] = self.decode_round_results({"pose_latent": output2_pose_latent, "img_latent": output2_img_latent}, height, width, apply_smaller_pose)
                output1_pose_latent = output2_pose_latent
                output1_img_latent = output2_img_latent
            return round_results, self.clip, self.t5, self.ae




    def forward_multiperson(self,
                            prompts: list[str],
                            general_prompt: str,
                            width: int = 512,
                            height: int = 512,
                            guidance: float = 4,
                            num_steps: int = 50,
                            seed: int = 123456789,
                            customize_img_ids: bool = False,
                            customize_txt_ids: bool = False,
                            bounding_boxes_in_order: list[list[int]] = None,
                            generate_img_ids_type: int = 1,
                            background_color: str = "black",
                            use_v1_bbox: bool = True, # if true, use previous v1-inference [bbox]
                            attn_mask_type: str = None, # None default -- if None, draw back to json2.py
                            embed_id: bool = False,
                            num_people_indicator: bool = False,
                            pretend_pe: bool = False,
                            pe_learnable: bool = False,
                            learned_shift_scale_ratio: float = 1,
                            ori_h: Tensor | None = None,
                            ori_w: Tensor | None = None,
                            apply_smaller_pose: int = 1,
                            ):
        round_results = {}
        noise_real = self.get_noise_patchified(height, width, seed) # [1, 1024, 64]
        # B, N, C = noise_real.shape
        # noise_pose = torch.clone(noise_real) # [1, 1024, 64]
        # noise_pose_unpacked = noise_pose.reshape(B, height//16, width//16, C)
        # noise_resize = noise_pose_unpacked[:, ::apply_smaller_pose, ::apply_smaller_pose, :]
        # noise_pose = noise_resize.reshape(B, -1, C)
        if apply_smaller_pose == 1:
            noise_pose = torch.clone(noise_real)
        else:
            noise_pose = self.get_noise_patchified(height//apply_smaller_pose, width//apply_smaller_pose, seed)

        black_latents = self.get_condition_latents_patchified(width//apply_smaller_pose, height//apply_smaller_pose, background_color) # [1, 1024, 64]

        # print("noise_pose.shape", noise_pose.shape)
        # print("black_latents.shape", black_latents.shape)
        # print("noise_real.shape", noise_real.shape)

        ##### deal for attn_mask_type #### do nothing but just full attention if attn_mask_type is None
        text_seqlen = 512
        cond_seqlen = black_latents.shape[1]
        pose_seqlen = noise_pose.shape[1]
        real_seqlen = noise_real.shape[1]
        batch_size = noise_pose.shape[0]
        if attn_mask_type is None:
            ar_attn_mask = None
        else:
            if attn_mask_type == "mask_condition_for_real":
                ar_attn_mask = self.create_attention_mask(batch_size=batch_size, text_seqlen=text_seqlen, cond_seqlen=cond_seqlen, pose_seqlen=pose_seqlen, real_seqlen=real_seqlen)
                ar_attn_mask = ar_attn_mask.to(noise_pose.device)
            else:
                raise ValueError(f"Invalid attn_mask_type: {attn_mask_type}")

        #############################################################################################

        timesteps = get_schedule(
            num_steps,
            # (width // 8) * (height // 8) // (16 * 16),
            (width // 16) * (height // 16),
            shift=True,
        )
        torch.manual_seed(seed)

        # first round
        print("first round")
        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            if num_people_indicator:
                input_prompt = ["Generate the first person of the scene: # " + prompts[0]]
            else:
                input_prompt = [prompts[0]] if not customize_txt_ids else [[prompts[0]]]
            if use_v1_bbox:
                if bounding_boxes_in_order is not None:
                    bounding_boxes_in_order.reverse()
                    input_bounding_boxes = [bounding_boxes_in_order]
                else:
                    input_bounding_boxes = None
            else:
                input_bounding_boxes = [[bounding_boxes_in_order[0]]] if bounding_boxes_in_order is not None else None
            inp_cond = self.prepare_multi_person(input_prompt, general_prompt, width, height, customize_img_ids, input_bounding_boxes, customize_txt_ids, generate_img_ids_type, embed_id, False, False, apply_smaller_pose)
            x_t = torch.cat([black_latents, noise_pose, noise_real], dim=1)
            visual_ids = torch.cat([inp_cond['img_ids_condition'], inp_cond['img_ids_pose'], inp_cond['img_ids_real']], dim=1)
            if embed_id:
                bbox_ids = torch.cat([inp_cond['bbox_ids_condition'], inp_cond['bbox_ids_pose'], inp_cond['bbox_ids_real']], dim=1)
            else:
                bbox_ids = None
            guidance_vec = torch.full((x_t.shape[0],), guidance, device=x_t.device, dtype=x_t.dtype)
            # print("DEBUG -----------------------------------")
            # print("x_t.shape", x_t.shape)
            # print("visual_ids.shape", visual_ids.shape)
            # print("guidance_vec.shape", guidance_vec.shape)
            # print("DEBUG -----------------------------------")
            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)

            real_seqlen = noise_real.shape[1]
            assert black_latents.shape[1] == noise_pose.shape[1]
            pose_seqlen = black_latents.shape[1]
            output1 = self.denoise_oneround(
                img=x_t,
                img_ids=visual_ids,
                bbox_ids=bbox_ids,
                txt=inp_cond['txt'],
                txt_ids=inp_cond['txt_ids'],
                vec=inp_cond['vec'],
                vec_pose=inp_cond['vec_pose'],
                timesteps=timesteps,
                guidance_vec=guidance_vec,
                image_sequence_len=real_seqlen,
                cond_sequence_len=pose_seqlen,
                ar_attn_mask=ar_attn_mask,
                embed_id=embed_id,
                pe_learnable=pe_learnable,
                learned_shift_scale_ratio=learned_shift_scale_ratio,
                batch_multi_person_last_indicator=torch.tensor([False], device=self.device),
            )
            #print("output1.shape", output1.shape)
            output1_pose_latent = output1[:, pose_seqlen:2*pose_seqlen, :]
            #print("output1_pose_latent.shape", output1_pose_latent.shape)
            output1_img_latent = output1[:, 2*pose_seqlen:, :]
            #print("output1_pose_latent.shape", output1_pose_latent.shape)
            #print("shape of [output1_pose_latent, output1_img_latent]", output1_pose_latent.shape, output1_img_latent.shape)
            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(output1_pose_latent.device)
            round_results["1"] = self.decode_round_results({"pose_latent": output1_pose_latent, "img_latent": output1_img_latent}, height, width, apply_smaller_pose)

            # second round
            print("second round")
            x_t = torch.cat([output1_pose_latent, noise_pose, noise_real], dim=1)
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)

            if num_people_indicator:
                input_prompt = ["Continue to generate the next person of the scene: " + prompts[0] + " # " + prompts[1]]
            else:
                input_prompt = [prompts[0] + " " + prompts[1]] if not customize_txt_ids else [[prompts[0], prompts[1]]]
            if use_v1_bbox:
                if bounding_boxes_in_order is not None:
                    bounding_boxes_in_order.reverse()
                    input_bounding_boxes = [bounding_boxes_in_order]
                else:
                    input_bounding_boxes = None
            else:
                input_bounding_boxes = [[bounding_boxes_in_order[0], bounding_boxes_in_order[1]]] if bounding_boxes_in_order is not None else None
            inp_cond = self.prepare_multi_person(input_prompt, general_prompt, width, height, customize_img_ids, input_bounding_boxes, customize_txt_ids, generate_img_ids_type, embed_id, pretend_pe, pe_learnable, apply_smaller_pose)
            visual_ids = torch.cat([inp_cond['img_ids_condition'], inp_cond['img_ids_pose'], inp_cond['img_ids_real']], dim=1)
            if embed_id:
                bbox_ids = torch.cat([inp_cond['bbox_ids_condition'], inp_cond['bbox_ids_pose'], inp_cond['bbox_ids_real']], dim=1)
            else:
                bbox_ids = None
            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)
            output2 = self.denoise_oneround(
                img=x_t,
                img_ids=visual_ids,
                bbox_ids=bbox_ids,
                txt=inp_cond['txt'],
                txt_ids=inp_cond['txt_ids'],
                vec=inp_cond['vec'],
                vec_pose=inp_cond['vec_pose'],
                timesteps=timesteps,
                guidance_vec=guidance_vec,
                image_sequence_len=real_seqlen,
                cond_sequence_len=pose_seqlen,
                ar_attn_mask=ar_attn_mask,
                embed_id=embed_id,
                pe_learnable=pe_learnable,
                learned_shift_scale_ratio=learned_shift_scale_ratio,
                batch_multi_person_last_indicator=torch.tensor([True], device=self.device),
                ori_h=ori_h,
                ori_w=ori_w,
            )
            output2_pose_latent = output2[:, pose_seqlen:2*pose_seqlen, :]
            output2_img_latent = output2[:, 2*pose_seqlen:, :]
            print("shape of [output2_pose_latent, output2_img_latent]", output2_pose_latent.shape, output2_img_latent.shape)
            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(output1_pose_latent.device)
            round_results["2"] = self.decode_round_results({"pose_latent": output2_pose_latent, "img_latent": output2_img_latent}, height, width, apply_smaller_pose)

            return round_results, self.clip, self.t5, self.ae

    def forward_multiperson_version2(self,
                            prompts: list[str],
                            general_prompt: str,
                            width: int = 512,
                            height: int = 512,
                            guidance: float = 4,
                            num_steps: int = 50,
                            seed: int = 123456789,
                            customize_img_ids: bool = False,
                            customize_txt_ids: bool = False,
                            bounding_boxes_in_order: list[list[int]] = None,
                            generate_img_ids_type: int = 1,
                            background_color: str = "black",
                            use_v1_bbox: bool = True, # if true, use previous v1-inference [bbox]
                            attn_mask_type: str = None, # None default -- if None, draw back to json2.py
                            embed_id: bool = False,
                            num_people_indicator: bool = False,
                            pretend_pe: bool = False,
                            pe_learnable: bool = False,
                            learned_shift_scale_ratio: float = 1,
                            ori_h: Tensor | None = None,
                            ori_w: Tensor | None = None,
                            apply_smaller_pose: int = 1,
                            ):
        round_results = {}
        noise_real = self.get_noise_patchified(height, width, seed) # [1, 1024, 64]
        # B, N, C = noise_real.shape
        # noise_pose = torch.clone(noise_real) # [1, 1024, 64]
        # noise_pose_unpacked = noise_pose.reshape(B, height//16, width//16, C)
        # noise_resize = noise_pose_unpacked[:, ::apply_smaller_pose, ::apply_smaller_pose, :]
        # noise_pose = noise_resize.reshape(B, -1, C)
        if apply_smaller_pose == 1:
            noise_pose = torch.clone(noise_real)
        else:
            noise_pose = self.get_noise_patchified(height//apply_smaller_pose, width//apply_smaller_pose, seed)

        black_latents = self.get_condition_latents_patchified(width//apply_smaller_pose, height//apply_smaller_pose, background_color) # [1, 1024, 64]
        white_latents = torch.zeros_like(black_latents) #self.get_white_latents_patchified(width//apply_smaller_pose, height//apply_smaller_pose) # [1, 1024, 64]

        # print("noise_pose.shape", noise_pose.shape)
        # print("black_latents.shape", black_latents.shape)
        # print("noise_real.shape", noise_real.shape)

        ##### deal for attn_mask_type #### do nothing but just full attention if attn_mask_type is None
        text_seqlen = 512
        cond_seqlen = black_latents.shape[1]
        pose_seqlen = noise_pose.shape[1]
        pre_real_seqlen = white_latents.shape[1]
        real_seqlen = noise_real.shape[1]
        batch_size = noise_pose.shape[0]
        if attn_mask_type is None:
            ar_attn_mask = None
        else:
            if attn_mask_type == "mask_condition_for_real":
                ar_attn_mask = self.create_attention_mask_version2(batch_size=batch_size, text_seqlen=text_seqlen, cond_seqlen=cond_seqlen, pose_seqlen=pose_seqlen, real_seqlen=real_seqlen)
                ar_attn_mask = ar_attn_mask.to(noise_pose.dtype)
                ar_attn_mask = ar_attn_mask.to(noise_pose.device)
            else:
                raise ValueError(f"Invalid attn_mask_type: {attn_mask_type}")

        #############################################################################################

        timesteps = get_schedule(
            num_steps,
            # (width // 8) * (height // 8) // (16 * 16),
            (width // 16) * (height // 16),
            shift=True,
        )
        torch.manual_seed(seed)

        # first round
        print("first round")
        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            if num_people_indicator:
                input_prompt = ["Generate the first person of the scene: # " + prompts[0]]
            else:
                input_prompt = [prompts[0]] if not customize_txt_ids else [[prompts[0]]]
            if use_v1_bbox:
                assert False, "don't use use_v1_bbox, use the normal one!"
                if bounding_boxes_in_order is not None:
                    bounding_boxes_in_order.reverse()
                    input_bounding_boxes = [bounding_boxes_in_order]
                else:
                    input_bounding_boxes = None
            else:
                input_bounding_boxes = [[bounding_boxes_in_order[0]]] if bounding_boxes_in_order is not None else None
            inp_cond = self.prepare_multi_person_version2(input_prompt, general_prompt, width, height, customize_img_ids, input_bounding_boxes, customize_txt_ids, generate_img_ids_type, embed_id, False, False, apply_smaller_pose)
            x_t = torch.cat([black_latents, noise_pose, white_latents, noise_real], dim=1)
            visual_ids = torch.cat([inp_cond['img_ids_condition'], inp_cond['img_ids_pose'], inp_cond["img_ids_pre_real"], inp_cond['img_ids_real']], dim=1)
            bbox_ids = None
            guidance_vec = torch.full((x_t.shape[0],), guidance, device=x_t.device, dtype=x_t.dtype)
            # print("DEBUG -----------------------------------")
            # print("x_t.shape", x_t.shape)
            # print("visual_ids.shape", visual_ids.shape)
            # print("guidance_vec.shape", guidance_vec.shape)
            # print("DEBUG -----------------------------------")
            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)

            real_seqlen = noise_real.shape[1]
            assert black_latents.shape[1] == noise_pose.shape[1]
            pose_seqlen = black_latents.shape[1]
            output1 = self.denoise_oneround_version2(
                img=x_t,
                img_ids=visual_ids,
                bbox_ids=bbox_ids,
                txt=inp_cond['txt'],
                txt_ids=inp_cond['txt_ids'],
                vec=inp_cond['vec'],
                vec_pose=inp_cond['vec_pose'],
                timesteps=timesteps, # all timesteps eg. 50 for sampling process
                guidance_vec=guidance_vec,
                image_sequence_len=real_seqlen,
                cond_sequence_len=pose_seqlen,
                ar_attn_mask=ar_attn_mask,
                embed_id=embed_id,
                pe_learnable=pe_learnable,
                learned_shift_scale_ratio=learned_shift_scale_ratio,
                batch_multi_person_last_indicator=torch.tensor([False], device=self.device),
            )
            #print("output1.shape", output1.shape)
            output1_pose_latent = output1[:, pose_seqlen:2*pose_seqlen, :]
            #print("output1_pose_latent.shape", output1_pose_latent.shape)
            output1_img_latent = output1[:, 2*pose_seqlen+real_seqlen:, :]
            #print("output1_pose_latent.shape", output1_pose_latent.shape)
            #print("shape of [output1_pose_latent, output1_img_latent]", output1_pose_latent.shape, output1_img_latent.shape)
            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(output1_pose_latent.device)
            round_results["1"] = self.decode_round_results({"pose_latent": output1_pose_latent, "img_latent": output1_img_latent}, height, width, apply_smaller_pose)

            # first_round_image = round_results["1"]["real_img"]
            # segmented_image = segment_on_white(first_round_image, bounding_boxes_in_order[0], self.device)
            # round_results["1"]['segmented_image'] = segmented_image
            pre_image_latent = output1_img_latent #pred_intermediate[:, 2*pose_seqlen+real_seqlen:, :] #output1_img_latent #self.get_segmented_latents_patchified(segmented_image)

            # second round
            print("second round")
            noise_real = self.get_noise_patchified(height, width, seed+1) # [1, 1024, 64]
            # B, N, C = noise_real.shape
            # noise_pose = torch.clone(noise_real) # [1, 1024, 64]
            # noise_pose_unpacked = noise_pose.reshape(B, height//16, width//16, C)
            # noise_resize = noise_pose_unpacked[:, ::apply_smaller_pose, ::apply_smaller_pose, :]
            # noise_pose = noise_resize.reshape(B, -1, C)
            if apply_smaller_pose == 1:
                noise_pose = torch.clone(noise_real)
            else:
                noise_pose = self.get_noise_patchified(height//apply_smaller_pose, width//apply_smaller_pose, seed+1)


            x_t = torch.cat([output1_pose_latent, noise_pose, pre_image_latent, noise_real], dim=1)
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)

            if num_people_indicator:
                input_prompt = ["Continue to generate the next person of the scene: " + prompts[0] + " # " + prompts[1]]
            else:
                input_prompt = [prompts[0] + " " + prompts[1]] if not customize_txt_ids else [[prompts[0], prompts[1]]]
            if use_v1_bbox:
                if bounding_boxes_in_order is not None:
                    bounding_boxes_in_order.reverse()
                    input_bounding_boxes = [bounding_boxes_in_order]
                else:
                    input_bounding_boxes = None
            else:
                input_bounding_boxes = [[bounding_boxes_in_order[0], bounding_boxes_in_order[1]]] if bounding_boxes_in_order is not None else None
            inp_cond = self.prepare_multi_person_version2(input_prompt, general_prompt, width, height, customize_img_ids, input_bounding_boxes, customize_txt_ids, generate_img_ids_type, embed_id, pretend_pe, pe_learnable, apply_smaller_pose)
            visual_ids = torch.cat([inp_cond['img_ids_condition'], inp_cond['img_ids_pose'], inp_cond["img_ids_pre_real"], inp_cond['img_ids_real']], dim=1)
            bbox_ids = None
            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)
            guidance_vec = torch.full((x_t.shape[0],), 3.5, device=x_t.device, dtype=x_t.dtype)
            output2 = self.denoise_oneround_version2(
                img=x_t,
                img_ids=visual_ids,
                bbox_ids=bbox_ids,
                txt=inp_cond['txt'],
                txt_ids=inp_cond['txt_ids'],
                vec=inp_cond['vec'],
                vec_pose=inp_cond['vec_pose'],
                timesteps=timesteps,
                guidance_vec=guidance_vec,
                image_sequence_len=real_seqlen,
                cond_sequence_len=pose_seqlen,
                ar_attn_mask=ar_attn_mask,
                embed_id=embed_id,
                pe_learnable=pe_learnable,
                learned_shift_scale_ratio=learned_shift_scale_ratio,
                batch_multi_person_last_indicator=torch.tensor([True], device=self.device),
                ori_h=ori_h,
                ori_w=ori_w,
            )
            output2_pose_latent = output2[:, pose_seqlen:2*pose_seqlen, :]
            output2_img_latent = output2[:,  2*pose_seqlen+real_seqlen:, :]
            print("shape of [output2_pose_latent, output2_img_latent]", output2_pose_latent.shape, output2_img_latent.shape)
            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(output1_pose_latent.device)
            round_results["2"] = self.decode_round_results({"pose_latent": output2_pose_latent, "img_latent": output2_img_latent}, height, width, apply_smaller_pose)

            return round_results, self.clip, self.t5, self.ae

    def forward_multiperson_version2_noseg(self,
                            prompts: list[str],
                            general_prompt: str,
                            width: int = 512,
                            height: int = 512,
                            guidance: float = 4,
                            num_steps: int = 50,
                            seed: int = 123456789,
                            customize_img_ids: bool = False,
                            customize_txt_ids: bool = False,
                            bounding_boxes_in_order: list[list[int]] = None,
                            generate_img_ids_type: int = 1,
                            background_color: str = "black",
                            use_v1_bbox: bool = True, # if true, use previous v1-inference [bbox]
                            attn_mask_type: str = None, # None default -- if None, draw back to json2.py
                            embed_id: bool = False,
                            num_people_indicator: bool = False,
                            pretend_pe: bool = False,
                            pe_learnable: bool = False,
                            learned_shift_scale_ratio: float = 1,
                            ori_h: Tensor | None = None,
                            ori_w: Tensor | None = None,
                            apply_smaller_pose: int = 1,
                            ):
        round_results = {}
        noise_real = self.get_noise_patchified(height, width, seed) # [1, 1024, 64]
        # B, N, C = noise_real.shape
        # noise_pose = torch.clone(noise_real) # [1, 1024, 64]
        # noise_pose_unpacked = noise_pose.reshape(B, height//16, width//16, C)
        # noise_resize = noise_pose_unpacked[:, ::apply_smaller_pose, ::apply_smaller_pose, :]
        # noise_pose = noise_resize.reshape(B, -1, C)
        if apply_smaller_pose == 1:
            noise_pose = torch.clone(noise_real)
        else:
            noise_pose = self.get_noise_patchified(height//apply_smaller_pose, width//apply_smaller_pose, seed)

        black_latents = self.get_condition_latents_patchified(width//apply_smaller_pose, height//apply_smaller_pose, background_color) # [1, 1024, 64]

        # print("noise_pose.shape", noise_pose.shape)
        # print("black_latents.shape", black_latents.shape)
        # print("noise_real.shape", noise_real.shape)

        ##### deal for attn_mask_type #### do nothing but just full attention if attn_mask_type is None
        text_seqlen = 512
        cond_seqlen = black_latents.shape[1]
        pose_seqlen = noise_pose.shape[1]
        real_seqlen = noise_real.shape[1]
        batch_size = noise_pose.shape[0]
        if attn_mask_type is None:
            ar_attn_mask = None
        else:
            if attn_mask_type == "mask_condition_for_real":
                ar_attn_mask = self.create_attention_mask(batch_size=batch_size, text_seqlen=text_seqlen, cond_seqlen=cond_seqlen, pose_seqlen=pose_seqlen, real_seqlen=real_seqlen)
                ar_attn_mask = ar_attn_mask.to(noise_pose.device)
                ar_attn_mask = ar_attn_mask.to(noise_pose.dtype)
            else:
                raise ValueError(f"Invalid attn_mask_type: {attn_mask_type}")

        #############################################################################################

        timesteps = get_schedule(
            num_steps,
            # (width // 8) * (height // 8) // (16 * 16),
            (width // 16) * (height // 16),
            shift=True,
        )
        torch.manual_seed(seed)

        # first round
        print("first round")
        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            if num_people_indicator:
                input_prompt = ["Generate the first person of the scene: # " + prompts[0]]
            else:
                input_prompt = [prompts[0]] if not customize_txt_ids else [[prompts[0]]]
            if use_v1_bbox:
                if bounding_boxes_in_order is not None:
                    bounding_boxes_in_order.reverse()
                    input_bounding_boxes = [bounding_boxes_in_order]
                else:
                    input_bounding_boxes = None
            else:
                input_bounding_boxes = [[bounding_boxes_in_order[0]]] if bounding_boxes_in_order is not None else None
            inp_cond = self.prepare_multi_person(input_prompt, general_prompt, width, height, customize_img_ids, input_bounding_boxes, customize_txt_ids, generate_img_ids_type, embed_id, False, False, apply_smaller_pose)
            x_t = torch.cat([black_latents, noise_pose, noise_real], dim=1)
            visual_ids = torch.cat([inp_cond['img_ids_condition'], inp_cond['img_ids_pose'], inp_cond['img_ids_real']], dim=1)
            if embed_id:
                bbox_ids = torch.cat([inp_cond['bbox_ids_condition'], inp_cond['bbox_ids_pose'], inp_cond['bbox_ids_real']], dim=1)
            else:
                bbox_ids = None
            guidance_vec = torch.full((x_t.shape[0],), guidance, device=x_t.device, dtype=x_t.dtype)
            # print("DEBUG -----------------------------------")
            # print("x_t.shape", x_t.shape)
            # print("visual_ids.shape", visual_ids.shape)
            # print("guidance_vec.shape", guidance_vec.shape)
            # print("DEBUG -----------------------------------")
            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)

            real_seqlen = noise_real.shape[1]
            assert black_latents.shape[1] == noise_pose.shape[1]
            pose_seqlen = black_latents.shape[1]
            output1 = self.denoise_oneround(
                img=x_t,
                img_ids=visual_ids,
                bbox_ids=bbox_ids,
                txt=inp_cond['txt'],
                txt_ids=inp_cond['txt_ids'],
                vec=inp_cond['vec'],
                vec_pose=inp_cond['vec_pose'],
                timesteps=timesteps,
                guidance_vec=guidance_vec,
                image_sequence_len=real_seqlen,
                cond_sequence_len=pose_seqlen,
                ar_attn_mask=ar_attn_mask,
                embed_id=embed_id,
                pe_learnable=pe_learnable,
                learned_shift_scale_ratio=learned_shift_scale_ratio,
                batch_multi_person_last_indicator=torch.tensor([False], device=self.device),
                version1=True,
            )
            #print("output1.shape", output1.shape)
            output1_pose_latent = output1[:, pose_seqlen:2*pose_seqlen, :]
            #print("output1_pose_latent.shape", output1_pose_latent.shape)
            output1_img_latent = output1[:, 2*pose_seqlen:, :]
            #print("output1_pose_latent.shape", output1_pose_latent.shape)
            #print("shape of [output1_pose_latent, output1_img_latent]", output1_pose_latent.shape, output1_img_latent.shape)
            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(output1_pose_latent.device)
            round_results["1"] = self.decode_round_results({"pose_latent": output1_pose_latent, "img_latent": output1_img_latent}, height, width, apply_smaller_pose)

            # first_round_image = round_results["1"]["real_img"]
            # segmented_image = segment_on_white(first_round_image, bounding_boxes_in_order[0], self.device)
            # round_results["1"]['segmented_image'] = segmented_image
            pre_image_latent = output1_img_latent #self.get_segmented_latents_patchified(first_round_image)

            # second round
            text_seqlen = 512
            cond_seqlen = black_latents.shape[1]
            pose_seqlen = noise_pose.shape[1]
            pre_real_seqlen = pre_image_latent.shape[1]
            real_seqlen = noise_real.shape[1]
            batch_size = noise_pose.shape[0]
            if attn_mask_type is None:
                ar_attn_mask = None
            else:
                if attn_mask_type == "mask_condition_for_real":
                    ar_attn_mask = self.create_attention_mask_version2(batch_size=batch_size, text_seqlen=text_seqlen, cond_seqlen=cond_seqlen, pose_seqlen=pose_seqlen, real_seqlen=real_seqlen)
                    ar_attn_mask = ar_attn_mask.to(noise_pose.dtype)
                    ar_attn_mask = ar_attn_mask.to(noise_pose.device)
                else:
                    raise ValueError(f"Invalid attn_mask_type: {attn_mask_type}")

            print("second round")

            x_t = torch.cat([output1_pose_latent, noise_pose, pre_image_latent, noise_real], dim=1)
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)

            if num_people_indicator:
                input_prompt = ["Continue to generate the next person of the scene: " + prompts[0] + " # " + prompts[1]]
            else:
                input_prompt = [prompts[0] + " " + prompts[1]] if not customize_txt_ids else [[prompts[0], prompts[1]]]
            if use_v1_bbox:
                if bounding_boxes_in_order is not None:
                    bounding_boxes_in_order.reverse()
                    input_bounding_boxes = [bounding_boxes_in_order]
                else:
                    input_bounding_boxes = None
            else:
                input_bounding_boxes = [[bounding_boxes_in_order[0], bounding_boxes_in_order[1]]] if bounding_boxes_in_order is not None else None
            inp_cond = self.prepare_multi_person_version2(input_prompt, general_prompt, width, height, customize_img_ids, input_bounding_boxes, customize_txt_ids, generate_img_ids_type, embed_id, pretend_pe, pe_learnable, apply_smaller_pose)
            visual_ids = torch.cat([inp_cond['img_ids_condition'], inp_cond['img_ids_pose'], inp_cond["img_ids_pre_real"], inp_cond['img_ids_real']], dim=1)
            bbox_ids = None
            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)
            output2 = self.denoise_oneround_version2(
                img=x_t,
                img_ids=visual_ids,
                bbox_ids=bbox_ids,
                txt=inp_cond['txt'],
                txt_ids=inp_cond['txt_ids'],
                vec=inp_cond['vec'],
                vec_pose=inp_cond['vec_pose'],
                timesteps=timesteps,
                guidance_vec=guidance_vec,
                image_sequence_len=real_seqlen,
                cond_sequence_len=pose_seqlen,
                ar_attn_mask=ar_attn_mask,
                embed_id=embed_id,
                pe_learnable=pe_learnable,
                learned_shift_scale_ratio=learned_shift_scale_ratio,
                batch_multi_person_last_indicator=torch.tensor([True], device=self.device),
                ori_h=ori_h,
                ori_w=ori_w,
            )
            output2_pose_latent = output2[:, pose_seqlen:2*pose_seqlen, :]
            output2_img_latent = output2[:,  2*pose_seqlen+real_seqlen:, :]
            print("shape of [output2_pose_latent, output2_img_latent]", output2_pose_latent.shape, output2_img_latent.shape)
            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(output1_pose_latent.device)
            round_results["2"] = self.decode_round_results({"pose_latent": output2_pose_latent, "img_latent": output2_img_latent}, height, width, apply_smaller_pose)

            return round_results, self.clip, self.t5, self.ae


    def forward_multiperson_version2_noseg_crossattn(self,
                            prompts: list[str],
                            general_prompt: str,
                            width: int = 512,
                            height: int = 512,
                            guidance: float = 4,
                            num_steps: int = 50,
                            seed: int = 123456789,
                            customize_img_ids: bool = False,
                            customize_txt_ids: bool = False,
                            bounding_boxes_in_order: list[list[int]] = None,
                            generate_img_ids_type: int = 1,
                            background_color: str = "black",
                            use_v1_bbox: bool = True, # if true, use previous v1-inference [bbox]
                            attn_mask_type: str = None, # None default -- if None, draw back to json2.py
                            embed_id: bool = False,
                            num_people_indicator: bool = False,
                            pretend_pe: bool = False,
                            pe_learnable: bool = False,
                            learned_shift_scale_ratio: float = 1,
                            ori_h: Tensor | None = None,
                            ori_w: Tensor | None = None,
                            apply_smaller_pose: int = 1,
                            ):
        round_results = {}
        noise_real = self.get_noise_patchified(height, width, seed) # [1, 1024, 64]
        # B, N, C = noise_real.shape
        # noise_pose = torch.clone(noise_real) # [1, 1024, 64]
        # noise_pose_unpacked = noise_pose.reshape(B, height//16, width//16, C)
        # noise_resize = noise_pose_unpacked[:, ::apply_smaller_pose, ::apply_smaller_pose, :]
        # noise_pose = noise_resize.reshape(B, -1, C)
        if apply_smaller_pose == 1:
            noise_pose = torch.clone(noise_real)
        else:
            noise_pose = self.get_noise_patchified(height//apply_smaller_pose, width//apply_smaller_pose, seed)

        black_latents = self.get_condition_latents_patchified(width//apply_smaller_pose, height//apply_smaller_pose, background_color) # [1, 1024, 64]

        # print("noise_pose.shape", noise_pose.shape)
        # print("black_latents.shape", black_latents.shape)
        # print("noise_real.shape", noise_real.shape)

        ##### deal for attn_mask_type #### do nothing but just full attention if attn_mask_type is None
        text_seqlen = 512
        cond_seqlen = black_latents.shape[1]
        pose_seqlen = noise_pose.shape[1]
        real_seqlen = noise_real.shape[1]
        batch_size = noise_pose.shape[0]
        if attn_mask_type is None:
            ar_attn_mask = None
        else:
            if attn_mask_type == "mask_condition_for_real":
                ar_attn_mask = self.create_attention_mask(batch_size=batch_size, text_seqlen=text_seqlen, cond_seqlen=cond_seqlen, pose_seqlen=pose_seqlen, real_seqlen=real_seqlen)
                ar_attn_mask = ar_attn_mask.to(noise_pose.device)
                ar_attn_mask = ar_attn_mask.to(noise_pose.dtype)
            else:
                raise ValueError(f"Invalid attn_mask_type: {attn_mask_type}")

        #############################################################################################

        timesteps = get_schedule(
            num_steps,
            # (width // 8) * (height // 8) // (16 * 16),
            (width // 16) * (height // 16),
            shift=True,
        )
        torch.manual_seed(seed)

        # first round
        print("first round")
        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            if num_people_indicator:
                input_prompt = ["Generate the first person of the scene: # " + prompts[0]]
            else:
                input_prompt = [prompts[0]] if not customize_txt_ids else [[prompts[0]]]
            if use_v1_bbox:
                if bounding_boxes_in_order is not None:
                    bounding_boxes_in_order.reverse()
                    input_bounding_boxes = [bounding_boxes_in_order]
                else:
                    input_bounding_boxes = None
            else:
                input_bounding_boxes = [[bounding_boxes_in_order[0]]] if bounding_boxes_in_order is not None else None
            inp_cond = self.prepare_multi_person(input_prompt, general_prompt, width, height, customize_img_ids, input_bounding_boxes, customize_txt_ids, generate_img_ids_type, embed_id, False, False, apply_smaller_pose)
            x_t = torch.cat([black_latents, noise_pose, noise_real], dim=1)
            visual_ids = torch.cat([inp_cond['img_ids_condition'], inp_cond['img_ids_pose'], inp_cond['img_ids_real']], dim=1)
            if embed_id:
                bbox_ids = torch.cat([inp_cond['bbox_ids_condition'], inp_cond['bbox_ids_pose'], inp_cond['bbox_ids_real']], dim=1)
            else:
                bbox_ids = None
            guidance_vec = torch.full((x_t.shape[0],), guidance, device=x_t.device, dtype=x_t.dtype)
            # print("DEBUG -----------------------------------")
            # print("x_t.shape", x_t.shape)
            # print("visual_ids.shape", visual_ids.shape)
            # print("guidance_vec.shape", guidance_vec.shape)
            # print("DEBUG -----------------------------------")
            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)

            real_seqlen = noise_real.shape[1]
            assert black_latents.shape[1] == noise_pose.shape[1]
            pose_seqlen = black_latents.shape[1]
            output1 = self.denoise_oneround(
                img=x_t,
                img_ids=visual_ids,
                bbox_ids=bbox_ids,
                txt=inp_cond['txt'],
                txt_ids=inp_cond['txt_ids'],
                vec=inp_cond['vec'],
                vec_pose=inp_cond['vec_pose'],
                timesteps=timesteps,
                guidance_vec=guidance_vec,
                image_sequence_len=real_seqlen,
                cond_sequence_len=pose_seqlen,
                ar_attn_mask=ar_attn_mask,
                embed_id=embed_id,
                pe_learnable=pe_learnable,
                learned_shift_scale_ratio=learned_shift_scale_ratio,
                batch_multi_person_last_indicator=torch.tensor([False], device=self.device),
                version1=True,
            )
            #print("output1.shape", output1.shape)
            output1_pose_latent = output1[:, pose_seqlen:2*pose_seqlen, :]
            #print("output1_pose_latent.shape", output1_pose_latent.shape)
            output1_img_latent = output1[:, 2*pose_seqlen:, :]
            #print("output1_pose_latent.shape", output1_pose_latent.shape)
            #print("shape of [output1_pose_latent, output1_img_latent]", output1_pose_latent.shape, output1_img_latent.shape)
            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(output1_pose_latent.device)
            round_results["1"] = self.decode_round_results({"pose_latent": output1_pose_latent, "img_latent": output1_img_latent}, height, width, apply_smaller_pose)

            # first_round_image = round_results["1"]["real_img"]
            # segmented_image = segment_on_white(first_round_image, bounding_boxes_in_order[0], self.device)
            # round_results["1"]['segmented_image'] = segmented_image
            pre_image_latent = output1_img_latent #self.get_segmented_latents_patchified(first_round_image)

            # second round
            text_seqlen = 512
            cond_seqlen = black_latents.shape[1]
            pose_seqlen = noise_pose.shape[1]
            pre_real_seqlen = pre_image_latent.shape[1]
            real_seqlen = noise_real.shape[1]
            batch_size = noise_pose.shape[0]
            if attn_mask_type is None:
                ar_attn_mask = None
            else:
                if attn_mask_type == "mask_condition_for_real":
                    ar_attn_mask = self.create_attention_mask_version2(batch_size=batch_size, text_seqlen=text_seqlen, cond_seqlen=cond_seqlen, pose_seqlen=pose_seqlen, real_seqlen=real_seqlen)
                    ar_attn_mask = ar_attn_mask.to(noise_pose.dtype)
                    ar_attn_mask = ar_attn_mask.to(noise_pose.device)
                else:
                    raise ValueError(f"Invalid attn_mask_type: {attn_mask_type}")

            print("second round")

            x_t = torch.cat([output1_pose_latent, noise_pose, pre_image_latent, noise_real], dim=1)
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)

            if num_people_indicator:
                input_prompt = ["Continue to generate the next person of the scene: " + prompts[0] + " # " + prompts[1]]
            else:
                input_prompt = [prompts[0] + " " + prompts[1]] if not customize_txt_ids else [[prompts[0], prompts[1]]]
            if use_v1_bbox:
                if bounding_boxes_in_order is not None:
                    bounding_boxes_in_order.reverse()
                    input_bounding_boxes = [bounding_boxes_in_order]
                else:
                    input_bounding_boxes = None
            else:
                input_bounding_boxes = [[bounding_boxes_in_order[0], bounding_boxes_in_order[1]]] if bounding_boxes_in_order is not None else None
            inp_cond = self.prepare_multi_person_version2(input_prompt, general_prompt, width, height, customize_img_ids, input_bounding_boxes, customize_txt_ids, generate_img_ids_type, embed_id, pretend_pe, pe_learnable, apply_smaller_pose)
            visual_ids = torch.cat([inp_cond['img_ids_condition'], inp_cond['img_ids_pose'], inp_cond["img_ids_pre_real"], inp_cond['img_ids_real']], dim=1)
            bbox_ids = None
            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)
            output2 = self.denoise_oneround_version2(
                img=x_t,
                img_ids=visual_ids,
                bbox_ids=bbox_ids,
                txt=inp_cond['txt'],
                txt_ids=inp_cond['txt_ids'],
                vec=inp_cond['vec'],
                vec_pose=inp_cond['vec_pose'],
                timesteps=timesteps,
                guidance_vec=guidance_vec,
                image_sequence_len=real_seqlen,
                cond_sequence_len=pose_seqlen,
                ar_attn_mask=ar_attn_mask,
                embed_id=embed_id,
                pe_learnable=pe_learnable,
                learned_shift_scale_ratio=learned_shift_scale_ratio,
                batch_multi_person_last_indicator=torch.tensor([True], device=self.device),
                ori_h=ori_h,
                ori_w=ori_w,
            )
            output2_pose_latent = output2[:, pose_seqlen:2*pose_seqlen, :]
            output2_img_latent = output2[:,  2*pose_seqlen+real_seqlen:, :]
            print("shape of [output2_pose_latent, output2_img_latent]", output2_pose_latent.shape, output2_img_latent.shape)
            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(output1_pose_latent.device)
            round_results["2"] = self.decode_round_results({"pose_latent": output2_pose_latent, "img_latent": output2_img_latent}, height, width, apply_smaller_pose)

            return round_results, self.clip, self.t5, self.ae


    def forward_multiperson_w_real_latent(self,
                            prompts: list[str],
                            general_prompt: str,
                            width: int = 512,
                            height: int = 512,
                            guidance: float = 4,
                            num_steps: int = 50,
                            seed: int = 123456789,
                            customize_img_ids: bool = False,
                            customize_txt_ids: bool = False,
                            bounding_boxes_in_order: list[list[int]] = None,
                            generate_img_ids_type: int = 1,
                            background_color: str = "black",
                            use_v1_bbox: bool = True, # if true, use previous v1-inference [bbox]
                            attn_mask_type: str = None, # None default -- if None, draw back to json2.py
                            embed_id: bool = False,
                            num_people_indicator: bool = False,
                            pretend_pe: bool = False,
                            pe_learnable: bool = False,
                            learned_shift_scale_ratio: float = 1,
                            ori_h: Tensor | None = None,
                            ori_w: Tensor | None = None,
                            apply_smaller_pose: int = 1,
                            ):
        round_results = {}
        noise_real = self.get_noise_patchified(height, width, seed) # [1, 1024, 64]
        # B, N, C = noise_real.shape
        # noise_pose = torch.clone(noise_real) # [1, 1024, 64]
        # noise_pose_unpacked = noise_pose.reshape(B, height//16, width//16, C)
        # noise_resize = noise_pose_unpacked[:, ::apply_smaller_pose, ::apply_smaller_pose, :]
        # noise_pose = noise_resize.reshape(B, -1, C)
        if apply_smaller_pose == 1:
            noise_pose = torch.clone(noise_real)
        else:
            noise_pose = self.get_noise_patchified(height//apply_smaller_pose, width//apply_smaller_pose, seed)

        black_latents = self.get_condition_latents_patchified(width//apply_smaller_pose, height//apply_smaller_pose, background_color) # [1, 1024, 64]

        # print("noise_pose.shape", noise_pose.shape)
        # print("black_latents.shape", black_latents.shape)
        # print("noise_real.shape", noise_real.shape)

        ##### deal for attn_mask_type #### do nothing but just full attention if attn_mask_type is None
        text_seqlen = 512
        cond_seqlen = black_latents.shape[1]
        pose_seqlen = noise_pose.shape[1]
        real_seqlen = noise_real.shape[1]
        batch_size = noise_pose.shape[0]
        if attn_mask_type is None:
            ar_attn_mask = None
        else:
            if attn_mask_type == "mask_condition_for_real":
                ar_attn_mask = self.create_attention_mask(batch_size=batch_size, text_seqlen=text_seqlen, cond_seqlen=cond_seqlen, pose_seqlen=pose_seqlen, real_seqlen=real_seqlen)
                ar_attn_mask = ar_attn_mask.to(noise_pose.device)
            else:
                raise ValueError(f"Invalid attn_mask_type: {attn_mask_type}")

        #############################################################################################

        timesteps = get_schedule(
            num_steps,
            # (width // 8) * (height // 8) // (16 * 16),
            (width // 16) * (height // 16),
            shift=True,
        )
        torch.manual_seed(seed)

        # first round
        print("first round")
        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            if num_people_indicator:
                input_prompt = ["Generate the first person of the scene: # " + prompts[0]]
            else:
                input_prompt = [prompts[0]] if not customize_txt_ids else [[prompts[0]]]
            if use_v1_bbox:
                if bounding_boxes_in_order is not None:
                    bounding_boxes_in_order.reverse()
                    input_bounding_boxes = [bounding_boxes_in_order]
                else:
                    input_bounding_boxes = None
            else:
                input_bounding_boxes = [[bounding_boxes_in_order[0]]] if bounding_boxes_in_order is not None else None
            inp_cond = self.prepare_multi_person(input_prompt, general_prompt, width, height, customize_img_ids, input_bounding_boxes, customize_txt_ids, generate_img_ids_type, embed_id, False, False, apply_smaller_pose)
            x_t = torch.cat([black_latents, noise_pose, noise_real], dim=1)
            visual_ids = torch.cat([inp_cond['img_ids_condition'], inp_cond['img_ids_pose'], inp_cond['img_ids_real']], dim=1)
            if embed_id:
                bbox_ids = torch.cat([inp_cond['bbox_ids_condition'], inp_cond['bbox_ids_pose'], inp_cond['bbox_ids_real']], dim=1)
            else:
                bbox_ids = None
            guidance_vec = torch.full((x_t.shape[0],), guidance, device=x_t.device, dtype=x_t.dtype)
            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)

            real_seqlen = noise_real.shape[1]
            assert black_latents.shape[1] == noise_pose.shape[1]
            pose_seqlen = black_latents.shape[1]
            output1 = self.denoise_oneround(
                img=x_t,
                img_ids=visual_ids,
                bbox_ids=bbox_ids,
                txt=inp_cond['txt'],
                txt_ids=inp_cond['txt_ids'],
                vec=inp_cond['vec'],
                vec_pose=inp_cond['vec_pose'],
                timesteps=timesteps,
                guidance_vec=guidance_vec,
                image_sequence_len=real_seqlen,
                cond_sequence_len=pose_seqlen,
                ar_attn_mask=ar_attn_mask,
                embed_id=embed_id,
                pe_learnable=pe_learnable,
                learned_shift_scale_ratio=learned_shift_scale_ratio,
                batch_multi_person_last_indicator=torch.tensor([False], device=self.device),
            )
            #print("output1.shape", output1.shape)
            output1_pose_latent = output1[:, pose_seqlen:2*pose_seqlen, :]
            #print("output1_pose_latent.shape", output1_pose_latent.shape)
            output1_img_latent = output1[:, 2*pose_seqlen:, :]
            #print("output1_pose_latent.shape", output1_pose_latent.shape)
            #print("shape of [output1_pose_latent, output1_img_latent]", output1_pose_latent.shape, output1_img_latent.shape)
            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(output1_pose_latent.device)
            round_results["1"] = self.decode_round_results({"pose_latent": output1_pose_latent, "img_latent": output1_img_latent}, height, width, apply_smaller_pose)

            # second round
            print("second round")
            start_step_idx = 3
            timestep_to_start_second_round = torch.tensor([timesteps[start_step_idx]], device=self.device)
            # x_t_pose = (1 - timestep_to_start_second_round[:, None, None]) * output1_pose_latent + timestep_to_start_second_round[:, None, None] * noise_pose
            x_t_real = (1 - timestep_to_start_second_round[:, None, None]) * output1_img_latent + timestep_to_start_second_round[:, None, None] * noise_real

            x_t = torch.cat([output1_pose_latent, noise_pose, x_t_real], dim=1)
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)

            if num_people_indicator:
                input_prompt = ["Continue to generate the next person of the scene: " + prompts[0] + " # " + prompts[1]]
            else:
                input_prompt = [prompts[0] + " " + prompts[1]] if not customize_txt_ids else [[prompts[0], prompts[1]]]
            if use_v1_bbox:
                if bounding_boxes_in_order is not None:
                    bounding_boxes_in_order.reverse()
                    input_bounding_boxes = [bounding_boxes_in_order]
                else:
                    input_bounding_boxes = None
            else:
                input_bounding_boxes = [[bounding_boxes_in_order[0], bounding_boxes_in_order[1]]] if bounding_boxes_in_order is not None else None
            inp_cond = self.prepare_multi_person(input_prompt, general_prompt, width, height, customize_img_ids, input_bounding_boxes, customize_txt_ids, generate_img_ids_type, embed_id, pretend_pe, pe_learnable, apply_smaller_pose)
            visual_ids = torch.cat([inp_cond['img_ids_condition'], inp_cond['img_ids_pose'], inp_cond['img_ids_real']], dim=1)
            if embed_id:
                bbox_ids = torch.cat([inp_cond['bbox_ids_condition'], inp_cond['bbox_ids_pose'], inp_cond['bbox_ids_real']], dim=1)
            else:
                bbox_ids = None
            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)
            output2 = self.denoise_oneround(
                img=x_t,
                img_ids=visual_ids,
                bbox_ids=bbox_ids,
                txt=inp_cond['txt'],
                txt_ids=inp_cond['txt_ids'],
                vec=inp_cond['vec'],
                vec_pose=inp_cond['vec_pose'],
                timesteps=timesteps[start_step_idx:],
                guidance_vec=guidance_vec,
                image_sequence_len=real_seqlen,
                cond_sequence_len=pose_seqlen,
                ar_attn_mask=ar_attn_mask,
                embed_id=embed_id,
                pe_learnable=pe_learnable,
                learned_shift_scale_ratio=learned_shift_scale_ratio,
                batch_multi_person_last_indicator=torch.tensor([True], device=self.device),
                ori_h=ori_h,
                ori_w=ori_w,
            )
            output2_pose_latent = output2[:, pose_seqlen:2*pose_seqlen, :]
            output2_img_latent = output2[:, 2*pose_seqlen:, :]
            print("shape of [output2_pose_latent, output2_img_latent]", output2_pose_latent.shape, output2_img_latent.shape)
            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(output1_pose_latent.device)
            round_results["2"] = self.decode_round_results({"pose_latent": output2_pose_latent, "img_latent": output2_img_latent}, height, width, apply_smaller_pose)

            return round_results, self.clip, self.t5, self.ae

    def forward_multiperson_firstround_only(self,
                            prompts: list[str],
                            general_prompt: str,
                            width: int = 512,
                            height: int = 512,
                            guidance: float = 4,
                            num_steps: int = 50,
                            seed: int = 123456789,
                            customize_img_ids: bool = False,
                            customize_txt_ids: bool = False,
                            bounding_boxes_in_order: list[list[int]] = None,
                            generate_img_ids_type: int = 1,
                            background_color: str = "black",
                            use_v1_bbox: bool = True, # if true, use previous v1-inference [bbox]
                            attn_mask_type: str = None, # None default -- if None, draw back to json2.py
                            embed_id: bool = False,
                            num_people_indicator: bool = False,
                            pretend_pe: bool = False,
                            ):
        round_results = {}
        noise_pose = self.get_noise_patchified(height, width, seed) # [1, 1024, 64]
        noise_real = torch.clone(noise_pose) # [1, 1024, 64]
        black_latents = self.get_condition_latents_patchified(width, height, background_color) # [1, 1024, 64]

        ##### deal for attn_mask_type #### do nothing but just full attention if attn_mask_type is None
        text_seqlen = 512
        cond_seqlen = black_latents.shape[1]
        pose_seqlen = noise_pose.shape[1]
        real_seqlen = noise_real.shape[1]
        batch_size = noise_pose.shape[0]
        if attn_mask_type is None:
            ar_attn_mask = None
        else:
            if attn_mask_type == "mask_condition_for_real":
                ar_attn_mask = self.create_attention_mask(batch_size=batch_size, text_seqlen=text_seqlen, cond_seqlen=cond_seqlen, pose_seqlen=pose_seqlen, real_seqlen=real_seqlen)
                ar_attn_mask = ar_attn_mask.to(noise_pose.device)
            else:
                raise ValueError(f"Invalid attn_mask_type: {attn_mask_type}")

        #############################################################################################

        timesteps = get_schedule(
            num_steps,
            # (width // 8) * (height // 8) // (16 * 16),
            (width // 16) * (height // 16),
            shift=True,
        )
        torch.manual_seed(seed)

        # first round
        print("first round")
        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            if num_people_indicator:
                input_prompt = ["Generate the first person of the scene: # " + prompts[0]]
            else:
                input_prompt = [prompts[0]] if not customize_txt_ids else [[prompts[0]]]
            if use_v1_bbox:
                if bounding_boxes_in_order is not None:
                    bounding_boxes_in_order.reverse()
                    input_bounding_boxes = [bounding_boxes_in_order]
                else:
                    input_bounding_boxes = None
            else:
                input_bounding_boxes = [[bounding_boxes_in_order[0]]] if bounding_boxes_in_order is not None else None
            inp_cond = self.prepare_multi_person(input_prompt, general_prompt, width, height, customize_img_ids, input_bounding_boxes, customize_txt_ids, generate_img_ids_type, embed_id)
            x_t = torch.cat([black_latents, noise_pose, noise_real], dim=1)
            visual_ids = torch.cat([inp_cond['img_ids_condition'], inp_cond['img_ids_pose'], inp_cond['img_ids_real']], dim=1)
            if embed_id:
                bbox_ids = torch.cat([inp_cond['bbox_ids_condition'], inp_cond['bbox_ids_pose'], inp_cond['bbox_ids_real']], dim=1)
            else:
                bbox_ids = None
            guidance_vec = torch.full((x_t.shape[0],), guidance, device=x_t.device, dtype=x_t.dtype)
            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)

            seqlen = (width // 16) * (height // 16)
            output1 = self.denoise_oneround(
                img=x_t,
                img_ids=visual_ids,
                bbox_ids=bbox_ids,
                txt=inp_cond['txt'],
                txt_ids=inp_cond['txt_ids'],
                vec=inp_cond['vec'],
                vec_pose=inp_cond['vec_pose'],
                timesteps=timesteps,
                guidance_vec=guidance_vec,
                image_sequence_len=seqlen,
                cond_sequence_len=seqlen,
                ar_attn_mask=ar_attn_mask,
                embed_id=embed_id,
            )

            output1_pose_latent = output1[:, seqlen:2*seqlen, :]
            output1_img_latent = output1[:, 2*seqlen:, :]
            print("shape of [output1_pose_latent, output1_img_latent]", output1_pose_latent.shape, output1_img_latent.shape)
            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(output1_pose_latent.device)
            round_results["1"] = self.decode_round_results({"pose_latent": output1_pose_latent, "img_latent": output1_img_latent}, height, width)

            return round_results, self.clip, self.t5, self.ae




    def forward_multiperson_embed_id_all_together(self,
                            prompts: list[str],
                            general_prompt: str,
                            width: int = 512,
                            height: int = 512,
                            guidance: float = 4,
                            num_steps: int = 50,
                            seed: int = 123456789,
                            customize_img_ids: bool = False,
                            customize_txt_ids: bool = False,
                            bounding_boxes_in_order: list[list[int]] = None,
                            generate_img_ids_type: int = 1,
                            background_color: str = "black",
                            use_v1_bbox: bool = True, # if true, use previous v1-inference [bbox]
                            attn_mask_type: str = None, # None default -- if None, draw back to json2.py
                            embed_id: bool = False,
                            ):
        round_results = {}
        noise_pose = self.get_noise_patchified(height, width, seed) # [1, 1024, 64]
        noise_real = torch.clone(noise_pose) # [1, 1024, 64]
        black_latents = self.get_condition_latents_patchified(width, height, background_color) # [1, 1024, 64]

        ##### deal for attn_mask_type #### do nothing but just full attention if attn_mask_type is None
        text_seqlen = 512
        cond_seqlen = black_latents.shape[1]
        pose_seqlen = noise_pose.shape[1]
        real_seqlen = noise_real.shape[1]
        batch_size = noise_pose.shape[0]
        if attn_mask_type is None:
            ar_attn_mask = None
        else:
            if attn_mask_type == "mask_condition_for_real":
                ar_attn_mask = self.create_attention_mask(batch_size=batch_size, text_seqlen=text_seqlen, cond_seqlen=cond_seqlen, pose_seqlen=pose_seqlen, real_seqlen=real_seqlen)
                ar_attn_mask = ar_attn_mask.to(noise_pose.device)
            else:
                raise ValueError(f"Invalid attn_mask_type: {attn_mask_type}")

        #############################################################################################

        timesteps = get_schedule(
            num_steps,
            # (width // 8) * (height // 8) // (16 * 16),
            (width // 16) * (height // 16),
            shift=True,
        )
        torch.manual_seed(seed)


        with torch.no_grad():

            seqlen = (width // 16) * (height // 16)
            # # second round
            print("second round")
            x_t = torch.cat([black_latents, noise_pose, noise_real], dim=1)
            guidance_vec = torch.full((x_t.shape[0],), guidance, device=x_t.device, dtype=x_t.dtype)
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            input_prompt = [prompts[0] + " " + prompts[1]] if not customize_txt_ids else [[prompts[0], prompts[1]]]
            if use_v1_bbox:
                if bounding_boxes_in_order is not None:
                    bounding_boxes_in_order.reverse()
                    input_bounding_boxes = [bounding_boxes_in_order]
                else:
                    input_bounding_boxes = None
            else:
                input_bounding_boxes = [[bounding_boxes_in_order[0], bounding_boxes_in_order[1]]] if bounding_boxes_in_order is not None else None
            inp_cond = self.prepare_multi_person(input_prompt, general_prompt, width, height, customize_img_ids, input_bounding_boxes, customize_txt_ids, generate_img_ids_type, embed_id)
            visual_ids = torch.cat([inp_cond['img_ids_condition'], inp_cond['img_ids_pose'], inp_cond['img_ids_real']], dim=1)
            if embed_id:
                bbox_ids = torch.cat([inp_cond['bbox_ids_condition'], inp_cond['bbox_ids_pose'], inp_cond['bbox_ids_real']], dim=1)
            else:
                bbox_ids = None

            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)
            output2 = self.denoise_oneround(
                img=x_t,
                img_ids=visual_ids,
                bbox_ids=bbox_ids,
                txt=inp_cond['txt'],
                txt_ids=inp_cond['txt_ids'],
                vec=inp_cond['vec'],
                vec_pose=inp_cond['vec_pose'],
                timesteps=timesteps,
                guidance_vec=guidance_vec,
                image_sequence_len=seqlen,
                cond_sequence_len=seqlen,
                ar_attn_mask=ar_attn_mask,
                embed_id=embed_id,
            )
            output2_pose_latent = output2[:, seqlen:2*seqlen, :]
            output2_img_latent = output2[:, 2*seqlen:, :]
            # print("shape of [output2_pose_latent, output2_img_latent]", output2_pose_latent.shape, output2_img_latent.shape)
            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(output2_pose_latent.device)
            round_results["2"] = self.decode_round_results({"pose_latent": output2_pose_latent, "img_latent": output2_img_latent}, height, width)

            round_results["1"] = round_results["2"]

            return round_results, self.clip, self.t5, self.ae



    def denoise_oneround(self, img, img_ids, bbox_ids, txt, txt_ids, vec, vec_pose, timesteps, guidance_vec, image_sequence_len, cond_sequence_len, ar_attn_mask=None, embed_id=False, pe_learnable=False, learned_shift_scale_ratio=1, batch_multi_person_last_indicator=None, ori_h=None, ori_w=None, version1=False):
        for t_curr, t_prev in tqdm(zip(timesteps[:-1], timesteps[1:])):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            t_cond = torch.full((img.shape[0],), timesteps[-1], dtype=img.dtype, device=img.device)

            pred = self.model(
                img=img,
                img_ids=img_ids,
                bbox_ids=bbox_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                y_pose=vec_pose,
                timesteps=t_vec,
                timesteps_pose=t_vec,
                timesteps_cond=t_cond,
                guidance=guidance_vec,
                image_sequence_len=image_sequence_len,
                cond_sequence_len=cond_sequence_len,
                ar_attn_mask=ar_attn_mask,
                embed_id=embed_id,
                pe_learnable=pe_learnable,
                learned_shift_scale_ratio=learned_shift_scale_ratio,
                batch_multi_person_last_indicator=batch_multi_person_last_indicator,
                ori_h=ori_h,
                ori_w=ori_w,
                version1=version1,
            )
            img = img + (t_prev - t_curr) * pred
        return img

    def denoise_oneround_oneshot(self, img, img_ids, bbox_ids, txt, txt_ids, vec, vec_pose, timesteps, guidance_vec, image_sequence_len, ar_attn_mask=None, embed_id=False, pe_learnable=False, learned_shift_scale_ratio=1, batch_multi_person_last_indicator=None, ori_h=None, ori_w=None, version1=False):
        for t_curr, t_prev in tqdm(zip(timesteps[:-1], timesteps[1:])):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            t_cond = torch.full((img.shape[0],), timesteps[-1], dtype=img.dtype, device=img.device)

            pred = self.model(
                img=img,
                img_ids=img_ids,
                bbox_ids=bbox_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                y_pose=vec_pose,
                timesteps=t_vec,
                timesteps_pose=t_vec,
                guidance=guidance_vec,
                image_sequence_len=image_sequence_len,
                ar_attn_mask=ar_attn_mask
            )
            img = img + (t_prev - t_curr) * pred
        return img

    def denoise_t2i(self, img, img_ids, bbox_ids, txt, txt_ids, vec, timesteps, guidance_vec, image_sequence_len, ar_attn_mask=None, embed_id=False, pe_learnable=False, learned_shift_scale_ratio=1, batch_multi_person_last_indicator=None, ori_h=None, ori_w=None, version1=False):
        for t_curr, t_prev in tqdm(zip(timesteps[:-1], timesteps[1:])):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            t_cond = torch.full((img.shape[0],), timesteps[-1], dtype=img.dtype, device=img.device)

            pred = self.model(
                img=img,
                img_ids=img_ids,
                bbox_ids=bbox_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                single_stream=True,
                image_sequence_len=image_sequence_len,
                ar_attn_mask=ar_attn_mask
            )
            img = img + (t_prev - t_curr) * pred
        return img

    def denoise_oneround_version2(self, img, img_ids, bbox_ids, txt, txt_ids, vec, vec_pose, timesteps, guidance_vec, image_sequence_len, cond_sequence_len, ar_attn_mask=None, embed_id=False, pe_learnable=False, learned_shift_scale_ratio=1, batch_multi_person_last_indicator=None, ori_h=None, ori_w=None):
        for t_curr, t_prev in tqdm(zip(timesteps[:-1], timesteps[1:])):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            t_cond = torch.full((img.shape[0],), timesteps[-1], dtype=img.dtype, device=img.device)

            pred = self.model(
                img=img,
                img_ids=img_ids,
                bbox_ids=bbox_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                y_pose=vec_pose,
                timesteps=t_vec,
                timesteps_pose=t_vec,
                timesteps_cond=t_cond,
                timesteps_pre_real=t_cond,
                guidance=guidance_vec,
                image_sequence_len=image_sequence_len,
                cond_sequence_len=cond_sequence_len,
                ar_attn_mask=ar_attn_mask,
                embed_id=embed_id,
                pe_learnable=pe_learnable,
                learned_shift_scale_ratio=learned_shift_scale_ratio,
                batch_multi_person_last_indicator=batch_multi_person_last_indicator,
                ori_h=ori_h,
                ori_w=ori_w,
            )
            img = img + (t_prev - t_curr) * pred
            # if t_curr == timesteps[30]:
            #     pred_intermediate = img.clone()
        return img #, pred_intermediate




    def __call__(self,
                 prompt: str,
                 image_prompt: Image = None,
                 controlnet_image: Image = None,
                 width: int = 512,
                 height: int = 512,
                 guidance: float = 4,
                 num_steps: int = 50,
                 seed: int = 123456789,
                 true_gs: float = 3,
                 control_weight: float = 0.9,
                 ip_scale: float = 1.0,
                 neg_ip_scale: float = 1.0,
                 neg_prompt: str = '',
                 neg_image_prompt: Image = None,
                 timestep_to_start_cfg: int = 0,
                 ## for pe, same as train
                 pose_id_value: float = 0.0,
                 add_h: bool = True,
                 use_attn_mask: bool = False,
                 ori_mode: bool = False,
                 add_pose_clip: bool = False,
                 ):
        width = 16 * (width // 16)
        height = 16 * (height // 16)
        image_proj = None
        neg_image_proj = None
        if not (image_prompt is None and neg_image_prompt is None) :
            assert self.ip_loaded, 'You must setup IP-Adapter to add image prompt as input'

            if image_prompt is None:
                image_prompt = np.zeros((width, height, 3), dtype=np.uint8)
            if neg_image_prompt is None:
                neg_image_prompt = np.zeros((width, height, 3), dtype=np.uint8)

            image_proj = self.get_image_proj(image_prompt)
            neg_image_proj = self.get_image_proj(neg_image_prompt)

        if self.controlnet_loaded:
            controlnet_image = self.annotator(controlnet_image, width, height)
            controlnet_image = torch.from_numpy((np.array(controlnet_image) / 127.5) - 1)
            controlnet_image = controlnet_image.permute(
                2, 0, 1).unsqueeze(0).to(torch.bfloat16).to(self.device)

        if ori_mode:
            return self.forward_ori(
                prompt,
                width,
                height,
                guidance,
                num_steps,
                seed,
                controlnet_image,
                timestep_to_start_cfg=timestep_to_start_cfg,
                true_gs=true_gs,
                control_weight=control_weight,
                neg_prompt=neg_prompt,
                image_proj=image_proj,
                neg_image_proj=neg_image_proj,
                ip_scale=ip_scale,
                neg_ip_scale=neg_ip_scale,
                use_attn_mask=use_attn_mask,
                pose_id_value=pose_id_value,
            )


        return self.forward(
            prompt,
            width,
            height,
            guidance,
            num_steps,
            seed,
            controlnet_image,
            timestep_to_start_cfg=timestep_to_start_cfg,
            true_gs=true_gs,
            control_weight=control_weight,
            neg_prompt=neg_prompt,
            image_proj=image_proj,
            neg_image_proj=neg_image_proj,
            ip_scale=ip_scale,
            neg_ip_scale=neg_ip_scale,
            pose_id_value=pose_id_value,
            add_h=add_h,
            use_attn_mask=use_attn_mask,
            add_pose_clip=add_pose_clip,
        )

    @torch.inference_mode()
    def gradio_generate(self, prompt, image_prompt, controlnet_image, width, height, guidance,
                        num_steps, seed, true_gs, ip_scale, neg_ip_scale, neg_prompt,
                        neg_image_prompt, timestep_to_start_cfg, control_type, control_weight,
                        lora_weight, local_path, lora_local_path, ip_local_path):
        if controlnet_image is not None:
            controlnet_image = Image.fromarray(controlnet_image)
            if ((self.controlnet_loaded and control_type != self.control_type)
                or not self.controlnet_loaded):
                if local_path is not None:
                    self.set_controlnet(control_type, local_path=local_path)
                else:
                    self.set_controlnet(control_type, local_path=None,
                                        repo_id=f"xlabs-ai/flux-controlnet-{control_type}-v3",
                                        name=f"flux-{control_type}-controlnet-v3.safetensors")
        if lora_local_path is not None:
            self.set_lora(local_path=lora_local_path, lora_weight=lora_weight)
        if image_prompt is not None:
            image_prompt = Image.fromarray(image_prompt)
            if neg_image_prompt is not None:
                neg_image_prompt = Image.fromarray(neg_image_prompt)
            if not self.ip_loaded:
                if ip_local_path is not None:
                    self.set_ip(local_path=ip_local_path)
                else:
                    self.set_ip(repo_id="xlabs-ai/flux-ip-adapter",
                                name="flux-ip-adapter.safetensors")
        seed = int(seed)
        if seed == -1:
            seed = torch.Generator(device="cpu").seed()

        img = self(prompt, image_prompt, controlnet_image, width, height, guidance,
                   num_steps, seed, true_gs, control_weight, ip_scale, neg_ip_scale, neg_prompt,
                   neg_image_prompt, timestep_to_start_cfg)

        filename = f"output/gradio/{uuid.uuid4()}.jpg"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Make] = "XLabs AI"
        exif_data[ExifTags.Base.Model] = self.model_type
        img.save(filename, format="jpeg", exif=exif_data, quality=95, subsampling=0)
        return img, filename

    def forward(
        self,
        prompt,
        width,
        height,
        guidance,
        num_steps,
        seed,
        controlnet_image = None,
        timestep_to_start_cfg = 0,
        true_gs = 3.5,
        control_weight = 0.9,
        neg_prompt="",
        image_proj=None,
        neg_image_proj=None,
        ip_scale=1.0,
        neg_ip_scale=1.0,
        pose_id_value=0.0,
        add_h=True,
        use_attn_mask=False,
        add_pose_clip=False,
    ):
        x_cat = get_noise(
            1, height*2, width, device=self.device,
            dtype=torch.bfloat16, seed=seed
        )
        # split by -2 dim
        x_pose = x_cat[:, :, :2 * math.ceil(height / 16), :]
        x = x_cat[:, :, 2 * math.ceil(height / 16):, :]

        timesteps = get_schedule(
            num_steps,
            # (width // 8) * (height // 8) // (16 * 16),
            (width // 16) * (height // 16),
            shift=True,
        )
        torch.manual_seed(seed)
        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            inp_cond = prepare_dual(t5=self.t5, clip=self.clip, img=x, prompt=prompt, img_pose=x_pose, pose_id_value=pose_id_value, add_h=add_h, add_pose_clip=add_pose_clip)
            neg_inp_cond = prepare_dual(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt, img_pose=x_pose, pose_id_value=pose_id_value, add_h=add_h)

            x_t = torch.cat([inp_cond['img_pose'], inp_cond['img']], dim=1)
            visual_ids = torch.cat([inp_cond['img_ids_pose'], inp_cond['img_ids']], dim=1)


            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)

            x = denoise(
                self.model,
                img=x_t,
                img_ids=visual_ids,
                txt=inp_cond['txt'],
                txt_ids=inp_cond['txt_ids'],
                vec=inp_cond['vec'],
                vec_pose=inp_cond['vec_pose'],
                timesteps=timesteps,
                guidance=guidance,
                timestep_to_start_cfg=timestep_to_start_cfg,
                neg_txt=neg_inp_cond['txt'],
                neg_txt_ids=neg_inp_cond['txt_ids'],
                neg_vec=neg_inp_cond['vec'],
                true_gs=true_gs,
                image_proj=image_proj,
                neg_image_proj=neg_image_proj,
                ip_scale=ip_scale,
                neg_ip_scale=neg_ip_scale,
                use_attn_mask=use_attn_mask
            )
            x_pose, x_img = x[:, :x.shape[1] // 2, :], x[:, x.shape[1] // 2:, :]

            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(x.device)
            x_pose = self.decode_x(x_pose, height, width)
            x_img = self.decode_x(x_img, height, width)

            x = torch.cat([x_pose, x_img], dim=-2)

        output_img = self.clamp_img(x)
        return output_img, self.clip, self.t5, self.ae

    def forward_dual_layout(
        self,
        prompt,
        clip_prompt,
        width,
        height,
        guidance,
        num_steps,
        seed,
        pose_id_value=0.0,
        add_pose_clip=False,
    ):
        noise_pose = self.get_noise_patchified(height, width, seed) # [1, 1024, 64]
        noise_real = torch.clone(noise_pose) # [1, 1024, 64]

        timesteps = get_schedule(
            num_steps,
            # (width // 8) * (height // 8) // (16 * 16),
            (width // 16) * (height // 16),
            shift=True,
        )
        torch.manual_seed(seed)
        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            inp_cond = prepare_dual_train_layout(t5=self.t5,
                                                 clip=self.clip,
                                                 bs=1,
                                                 h_pose=[height//8],
                                                 w_pose=[width//8],
                                                 h_real=[height//8],
                                                 w_real=[width//8],
                                                 device=self.device,
                                                 prompt=prompt,
                                                 clip_prompt=clip_prompt,
                                                 pose_id_value=pose_id_value,
                                                 add_pose_clip=add_pose_clip,
                                                 )

            x_t = torch.cat([noise_pose, noise_real], dim=1)
            visual_ids = torch.cat([inp_cond['img_ids_pose'], inp_cond['img_ids_real']], dim=1)


            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)

            x = denoise(
                self.model,
                img=x_t,
                img_ids=visual_ids,
                txt=inp_cond['txt'],
                txt_ids=inp_cond['txt_ids'],
                vec=inp_cond['vec'],
                vec_pose=inp_cond['vec_pose'],
                timesteps=timesteps,
                guidance=guidance,
            )
            x_pose, x_img = x[:, :x.shape[1] // 2, :], x[:, x.shape[1] // 2:, :]

            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(x.device)
            x_pose = self.decode_x(x_pose, height, width)
            x_img = self.decode_x(x_img, height, width)

            x = torch.cat([x_pose, x_img], dim=-2)

        output_img = self.clamp_img(x)
        return output_img, self.clip, self.t5, self.ae

    def decode_x(self, x, height, width):
        x = unpack(x.float(), height, width)
        # Move to same device as VAE decoder if offloaded
        if self.offload and x.device != next(self.ae.decoder.parameters()).device:
            x = x.to(next(self.ae.decoder.parameters()).device)
        x = self.ae.decode(x)
        return x

    def clamp_img(self, x):
        x1 = x.clamp(-1, 1)
        x1 = rearrange(x1[-1], "c h w -> h w c")
        output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
        return output_img

    def forward_ori(
        self,
        prompt,
        width,
        height,
        guidance,
        num_steps,
        seed,
        image_proj=None,
        ip_scale=1.0,
        pose_id_value=0.0,
        use_attn_mask=False,
    ):
        x = get_noise(
            1, height, width, device=self.device,
            dtype=torch.bfloat16, seed=seed
        )
        timesteps = get_schedule(
            num_steps,
            # (width // 8) * (height // 8) // (16 * 16),
            (width // 16) * (height // 16),
            shift=True,
        )
        torch.manual_seed(seed)
        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt, pose_id_value=pose_id_value)

            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)

            x = denoise(
                self.model,
                **inp_cond,
                timesteps=timesteps,
                guidance=guidance,
                image_proj=image_proj,
                ip_scale=ip_scale,
                use_attn_mask=use_attn_mask,
                single_stream=True,
            )

            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(x.device)
            x = unpack(x.float(), height, width)
            x = self.ae.decode(x)
            self.offload_model_to_cpu(self.ae.decoder)

        x1 = x.clamp(-1, 1)
        x1 = rearrange(x1[-1], "c h w -> h w c")
        output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
        return output_img, self.clip, self.t5, self.ae

    def forward_ori_split_resolution(
        self,
        prompt,
        width1,
        height1,
        width2,
        height2,
        guidance,
        num_steps,
        seed,
        image_proj=None,
        ip_scale=1.0,
        pose_id_value=0.0,
        use_attn_mask=False,
        change_pe=False,
        smaller_scale=1,
        with_black_latents=False,
    ):
        if with_black_latents:
            x0 = create_black_image_latents(self.ae, height1, width1, self.device)
        x1 = get_noise(
            1, height1, width1, device=self.device,
            dtype=torch.bfloat16, seed=seed
        )
        x2 = get_noise(
            1, height2, width2, device=self.device,
            dtype=torch.bfloat16, seed=seed
        )
        timesteps = get_schedule(
            num_steps,
            # (width // 8) * (height // 8) // (16 * 16),
            (width2 // 16) * (height2 // 16),
            shift=True,
        )
        torch.manual_seed(seed)
        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            inp_cond = prepare_split_resolution(t5=self.t5, clip=self.clip, img1=x1, img2=x2, prompt=prompt, pose_id_value=pose_id_value, change_pe=change_pe, smaller_scale=smaller_scale, black_latents=x0)

            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)

            x = denoise(
                self.model,
                **inp_cond,
                timesteps=timesteps,
                guidance=guidance,
                image_proj=image_proj,
                ip_scale=ip_scale,
                use_attn_mask=use_attn_mask,
                single_stream=True,
            )

            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(x.device)

            x1_start = 0
            seqlen1 = (width1 // 16) * (height1 // 16)
            if with_black_latents:
                x1_start = seqlen1
            x1 = x[:, x1_start:x1_start+seqlen1, :]
            x2 = x[:, x1_start+seqlen1:, :]
            x1 = unpack(x1.float(), height1, width1)
            x2 = unpack(x2.float(), height2, width2)
            x1 = self.ae.decode(x1)
            x2 = self.ae.decode(x2)
            self.offload_model_to_cpu(self.ae.decoder)

        x1 = x1.clamp(-1, 1)
        x1 = rearrange(x1[-1], "c h w -> h w c")
        x2 = x2.clamp(-1, 1)
        x2 = rearrange(x2[-1], "c h w -> h w c")
        output_img1 = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
        output_img2 = Image.fromarray((127.5 * (x2 + 1.0)).cpu().byte().numpy())
        return output_img1, output_img2, self.clip, self.t5, self.ae



    def forward_ori_w_prompt_mask(
        self,
        prompt_list,
        clip_prompt,
        width,
        height,
        guidance,
        num_steps,
        seed,
        true_gs = 3.5,
        image_proj=None,
        ip_scale=1.0,
        use_attn_mask=False,
        ids_to_mask=[1,],
        doubleblock_apply_mask_from=0,
        apply_mask_from_denoise_step=0,
    ):
        x = get_noise(
            1, height, width, device=self.device,
            dtype=torch.bfloat16, seed=seed
        )
        batch_size, txt_len = 1, 512

        timesteps = get_schedule(
            num_steps,
            # (width // 8) * (height // 8) // (16 * 16),
            (width // 16) * (height // 16),
            shift=True,
        )
        torch.manual_seed(seed)
        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            inp_cond = prepare_w_prompt_mask(t5=self.t5, clip=self.clip, img=x, prompt_list=prompt_list, clip_prompt=clip_prompt)
            img_seqlen = inp_cond["img"].shape[1]
            # print(f"img_seqlen: {img_seqlen}")
            total_len = 512 + img_seqlen
            # print(f"total_len: {total_len}")
            txt_mask_ids = inp_cond["txt_mask_ids"] # [1, 512]
            # A value of False means the tokens can attend to each other.
            attention_mask = torch.zeros((batch_size, total_len, total_len), dtype=torch.bool, device=txt_mask_ids.device)

            # First, we create a boolean mask for the text token IDs that need to be masked.
            # This tensor has shape [batch_size, txt_len].
            masked_txt_tokens_bool = torch.isin(txt_mask_ids, torch.tensor(ids_to_mask, device=txt_mask_ids.device))
            img_tokens_false_mask = torch.zeros((batch_size, img_seqlen), dtype=torch.bool, device=txt_mask_ids.device)
            combined_col_mask = torch.cat((masked_txt_tokens_bool, img_tokens_false_mask), dim=1)

            # Now, we apply the mask.
            # The rows (queries) are the image tokens, which start at `txt_len`.
            # The columns (keys) are the text tokens that we want to mask.
            # `combined_col_mask` is used to index the columns correctly.
            # `True` is assigned to these positions to indicate masking.
            attention_mask[:, txt_len:, combined_col_mask.squeeze()] = True
            attention_mask = ~attention_mask
            # print("dtype of attention_mask: ", attention_mask.dtype)

            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)

            x = denoise(
                self.model,
                img=inp_cond["img"],
                img_ids=inp_cond["img_ids"],
                txt=inp_cond["txt"],
                txt_ids=inp_cond["txt_ids"],
                vec=inp_cond["vec"],
                timesteps=timesteps,
                guidance=guidance,
                true_gs=true_gs,
                image_proj=image_proj,
                ip_scale=ip_scale,
                use_attn_mask=use_attn_mask,
                single_stream=True,
                attention_mask=attention_mask,
                doubleblock_apply_mask_from=doubleblock_apply_mask_from,
                apply_mask_from_denoise_step=apply_mask_from_denoise_step,
            )

            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(x.device)
            x = unpack(x.float(), height, width)
            x = self.ae.decode(x)
            self.offload_model_to_cpu(self.ae.decoder)

        x1 = x.clamp(-1, 1)
        x1 = rearrange(x1[-1], "c h w -> h w c")
        output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
        return output_img, self.clip, self.t5, self.ae


    def offload_model_to_cpu(self, *models):
        if not self.offload: return
        for model in models:
            model.cpu()
            torch.cuda.empty_cache()


class XFluxSampler(XFluxPipeline):
    def __init__(self, clip, t5, ae, model, device, offload=True):
        if clip is None:
            self.clip = load_clip(device)
        else:
            self.clip = clip
        if t5 is None:
            self.t5 = load_t5(device, max_length=512)
        else:
            self.t5 = t5
        if ae is None:
            self.ae = load_ae("flux-dev", device=device)
        else:
            self.ae = ae
        self.model = model
        self.model.eval()
        self.device = device
        self.controlnet_loaded = False
        self.ip_loaded = False
        self.offload = offload

    def forward_ar(
        self,
        prompts,  # List of prompts, one per round
        general_prompt,
        width,
        height,
        num_steps,
        seed,
        timestep_to_start_cfg = 0,
        guidance: float = 4,
        true_gs = 3.5,
        neg_prompt="",
        image_proj=None,
        neg_image_proj=None,
        ip_scale=1.0,
        neg_ip_scale=1.0,
        pose_id_value=0.0,
        use_attn_mask=False,
        add_pose_clip=False,
        num_rounds=2,
    ):
        """
        Iterative generation: each prompt corresponds to one round.

        Flow:
        - Round 1: [black_latents, noise1_pose, noise1_real] → denoise → [output1_pose, output1_real]
        - Round 2: [output1_pose_latents, noise2_pose, noise2_real] → denoise → [output2_pose, output2_real]
        """
        timesteps = get_schedule(
            num_steps,
            # (width // 8) * (height // 8) // (16 * 16),
            (width // 16) * (height // 16),
            shift=True,
        )

        torch.manual_seed(seed)

        # Ensure we have enough prompts for the requested number of rounds
        if len(prompts) < num_rounds:
            prompts = prompts + [prompts[-1]] * (num_rounds - len(prompts))

        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)

            # Create black image latents for initial condition
            black_latents = create_black_image_latents(self.ae, height, width, self.device)

            # Sequence lengths (fixed for 512x512 images)
            cond_seq_len = 4096
            pose_seq_len = 4096
            real_seq_len = 4096

            all_rounds_output = []
            prev_pose_latents = black_latents  # Start with black latents

            for round_idx in range(num_rounds):
                current_prompt = prompts[round_idx]
                print(f"Starting iterative round {round_idx + 1}/{num_rounds} with prompt: {current_prompt}")

                # Generate noise for pose and real
                noise_pose = get_noise(1, height, width, self.device, torch.bfloat16, seed + round_idx)
                noise_real = get_noise(1, height, width, self.device, torch.bfloat16, seed + round_idx)

                # For first round, display initial noise images
                if round_idx == 0:
                    self._log_debug_noise_images(noise_pose, noise_real, height, width, round_idx)

                # Patchify noise
                noise_pose_patchified = rearrange(noise_pose, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
                noise_real_patchified = rearrange(noise_real, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

                # Concatenate: [condition, pose_noise, real_noise]
                x_t = torch.cat([prev_pose_latents.unsqueeze(0), noise_pose_patchified, noise_real_patchified], dim=1)

                # Prepare text embeddings for this round
                inp_cond = prepare_dual_train_ar(
                    t5=self.t5,
                    clip=self.clip,
                    bs=1,
                    h_condition=[height//8],
                    w_condition=[width//8],
                    h_pose=[height//8],
                    w_pose=[width//8],
                    h_real=[height//8],
                    w_real=[width//8],
                    device=self.device,
                    prompt=current_prompt,  # Use round-specific prompt for T5
                    clip_prompt=general_prompt,  # Use shared general_prompt for CLIP
                    pose_id_value=pose_id_value,
                    add_pose_clip=add_pose_clip
                )

                neg_inp_cond = prepare_dual_train_ar(
                    t5=self.t5,
                    clip=self.clip,
                    bs=1,
                    h_condition=[height//8],
                    w_condition=[width//8],
                    h_pose=[height//8],
                    w_pose=[height//8],
                    h_real=[height//8],
                    w_real=[width//8],
                    device=self.device,
                    prompt=neg_prompt,  # Negative prompt for T5
                    clip_prompt=general_prompt,  # Shared general_prompt for CLIP
                    pose_id_value=pose_id_value,
                    add_pose_clip=add_pose_clip
                )

                # Concatenate visual IDs
                visual_ids = torch.cat([inp_cond['img_ids_condition'], inp_cond['img_ids_pose'], inp_cond['img_ids_real']], dim=1)

                if self.offload:
                    self.offload_model_to_cpu(self.t5, self.clip)
                    self.model = self.model.to(self.device)

                # Denoise iteratively.
                x = denoise_ar(
                    self.model,
                    img=x_t,
                    img_ids=visual_ids,
                    txt=inp_cond['txt'],
                    txt_ids=inp_cond['txt_ids'],
                    vec=inp_cond['vec'],
                    vec_pose=inp_cond['vec_pose'],
                    timesteps=timesteps,
                    guidance=guidance,
                    timestep_to_start_cfg=timestep_to_start_cfg,
                    neg_txt=neg_inp_cond['txt'],
                    neg_txt_ids=neg_inp_cond['txt_ids'],
                    neg_vec=neg_inp_cond['vec'],
                    true_gs=true_gs,
                    image_proj=image_proj,
                    neg_image_proj=neg_image_proj,
                    ip_scale=ip_scale,
                    neg_ip_scale=neg_ip_scale,
                    use_attn_mask=use_attn_mask,
                    cond_sequence_len=cond_seq_len,
                    pose_sequence_len=pose_seq_len,
                )

                # Split outputs
                x_cond = x[:, :cond_seq_len, :]
                x_pose = x[:, cond_seq_len:cond_seq_len+pose_seq_len, :]
                x_real = x[:, cond_seq_len+pose_seq_len:, :]

                # Store pose latents for next round
                prev_pose_latents = x_pose.squeeze(0)

                # Decode images for visualization
                if self.offload:
                    self.offload_model_to_cpu(self.model)
                    self.ae.decoder.to(x.device)

                x_pose_decoded = self.decode_x(x_pose, height, width)
                x_real_decoded = self.decode_x(x_real, height, width)

                # Convert to PIL images for wandb logging
                x_pose_pil = self.clamp_img(x_pose_decoded)
                x_real_pil = self.clamp_img(x_real_decoded)

                # Combine pose and real images for display
                combined_img = torch.cat([x_pose_decoded, x_real_decoded], dim=-2)
                combined_pil = self.clamp_img(combined_img)

                all_rounds_output.append({
                    'round': round_idx + 1,
                    'prompt': current_prompt,
                    'pose_img': x_pose_pil,
                    'real_img': x_real_pil,
                    'combined_img': combined_pil,
                    'pose_latents': prev_pose_latents
                })

                print(f"Completed round {round_idx + 1}")

            if self.offload:
                self.offload_model_to_cpu(self.ae.decoder)

            return all_rounds_output, self.clip, self.t5, self.ae

    def _log_debug_noise_images(self, noise_pose, noise_real, height, width, round_idx):
        """Helper method to log debug noise images for the first round"""
        # Decode and display initial noise
        noise_pose_patchified = rearrange(noise_pose, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        noise_real_patchified = rearrange(noise_real, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

        # Move to same device as VAE decoder
        if self.offload:
            noise_pose_patchified = noise_pose_patchified.cpu()
            noise_real_patchified = noise_real_patchified.cpu()

        noise_pose_decoded = self.decode_x(noise_pose_patchified, height, width)
        noise_real_decoded = self.decode_x(noise_real_patchified, height, width)

        noise_pose_pil = self.clamp_img(noise_pose_decoded)
        noise_real_pil = self.clamp_img(noise_real_decoded)

        # Store for wandb logging (will be logged in sample_step)
        if not hasattr(self, 'debug_images'):
            self.debug_images = []
        self.debug_images.extend([
            {'type': 'initial_noise_pose', 'image': noise_pose_pil, 'round': round_idx + 1},
            {'type': 'initial_noise_real', 'image': noise_real_pil, 'round': round_idx + 1}
        ])

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch.nn as nn
from torch import Tensor
import torch
from omegaconf import OmegaConf
import copy
from src.flux.xflux_pipeline import XFluxSampler
from PIL import Image, ImageDraw, ImageFont
import tempfile
import random

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError

from src.flux.modules.layers import LoRALinearLayer, LastLayer
from src.flux.train_utils import replace_attn_processor_triplelora_ar
from src.flux.util import load_flow_model2

class ImgInSplit(nn.Module): # must call after loading pre-trained model
    def __init__(self, old_img_in: nn.Linear, keep_ori_weights: bool = False, zero_init: bool = False):
        super().__init__()
        assert not (keep_ori_weights and zero_init), "keep_ori_weights and zero_init cannot be both True"
        self.old_img_in = old_img_in

        self.pose_in = copy.deepcopy(old_img_in)
        if not keep_ori_weights:
            if zero_init:
                nn.init.zeros_(self.pose_in.weight)
                nn.init.zeros_(self.pose_in.bias)
            else:
                nn.init.normal_(self.pose_in.weight, mean=0.0, std=0.02)
                nn.init.zeros_(self.pose_in.bias)


    def forward(self, x: Tensor, img_seq_len: int) -> Tensor:
        assert x.dim() == 3, "x should be in shape (B, L1+L2, D)"
        B, L, D = x.shape
        pose_len = L - img_seq_len

        x_pose = x[:, :pose_len, :]
        x_img = x[:, pose_len:, :]

        x_img = self.old_img_in(x_img)
        x_pose = self.pose_in(x_pose)

        # print("DEBUG -----------------------------------")
        # print("after img_in x_pose.shape", x_pose.shape)
        # print("after img_in x_img.shape", x_img.shape)
        # print("DEBUG -----------------------------------")

        return torch.cat([x_pose, x_img], dim=1)

class LastLayerSplitTwoMod(nn.Module): # two vec condition version
    """
    Same math as the original LastLayer, but with
      • two independent output heads (linear1, linear2)
      • two independent AdaLN modulators (ada1, ada2)

    Args
    ----
    old_layer : a *loaded* LastLayer whose weights you want to duplicate.
    """

    def __init__(self, old_layer: "LastLayer", keep_ori_weights: bool = False, zero_init: bool = False):
        super().__init__()
        self.old_layer = old_layer

        # duplicate AdaLN MLPs
        self.adaLN_modulation_pose = copy.deepcopy(old_layer.adaLN_modulation)
        if not keep_ori_weights:
            if zero_init:
                nn.init.zeros_(self.adaLN_modulation_pose[1].weight)
                nn.init.zeros_(self.adaLN_modulation_pose[1].bias)
            else:
                nn.init.normal_(self.adaLN_modulation_pose[1].weight, mean=0.0, std=0.02)
                nn.init.zeros_(self.adaLN_modulation_pose[1].bias)

        # duplicate output heads
        self.linear_pose_img = copy.deepcopy(old_layer.linear)
        if not keep_ori_weights:
            if zero_init:
                nn.init.zeros_(self.linear_pose_img.weight)
                nn.init.zeros_(self.linear_pose_img.bias)
            else:
                nn.init.normal_(self.linear_pose_img.weight, mean=0.0, std=0.02)
                nn.init.zeros_(self.linear_pose_img.bias)

    # ---------------------------------------------------------------------
    def forward(self, x: Tensor, vec1: Tensor, vec2: Tensor, img_seq_len: int) -> Tensor:
        """
        x     : (B, 2*T, hidden_size)
        vec1  : (B, hidden_size)  – conditioning for the *first* half
        vec2  : (B, hidden_size)  – conditioning for the *second* half
        """
        assert x.dim() == 3, "x should be in shape (B, L1+L2, D)"
        B, L, D = x.shape
        pose_len = L - img_seq_len

        x_pose = x[:, :pose_len, :]
        x_img = x[:, pose_len:, :]

        # branch 1
        shift, scale = self.old_layer.adaLN_modulation(vec1).chunk(2, dim=1)
        x_img = (1 + scale[:, None, :]) * self.old_layer.norm_final(x_img) + shift[:, None, :]
        x_img = self.old_layer.linear(x_img)

        # branch 2
        shift_pose, scale_pose = self.adaLN_modulation_pose(vec2).chunk(2, dim=1)
        x_pose = (1 + scale_pose[:, None, :]) * self.old_layer.norm_final(x_pose) + shift_pose[:, None, :] # ERROR!
        x_pose = self.linear_pose_img(x_pose)

        return torch.cat([x_pose, x_img], dim=1)

    def forward_pose_only(self, x: Tensor, vec2: Tensor) -> Tensor:
        """
        x     : (B, 2*T, hidden_size)
        vec1  : (B, hidden_size)  – conditioning for the *first* half
        vec2  : (B, hidden_size)  – conditioning for the *second* half
        """
        assert x.dim() == 3, "x should be in shape (B, L1+L2, D)"
        x_pose = x

        # branch 2
        shift_pose, scale_pose = self.adaLN_modulation_pose(vec2).chunk(2, dim=1)
        x_pose = (1 + scale_pose[:, None, :]) * self.old_layer.norm_final(x_pose) + shift_pose[:, None, :] # ERROR!
        x_pose = self.linear_pose_img(x_pose)

        return x_pose


def replace_split_head(dit, args):
    old_img_in = dit.img_in
    dit.img_in = ImgInSplit(old_img_in, keep_ori_weights=args.keep_ori_weights, zero_init=args.zero_init)

    old_final_layer = dit.final_layer
    dit.final_layer = LastLayerSplitTwoMod(old_final_layer, keep_ori_weights=args.keep_ori_weights, zero_init=args.zero_init)


def _load_eval_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict JSON at top level, got {type(data)}")
    if "eval" not in data or not isinstance(data["eval"], dict):
        raise ValueError(
            "Expected JSON to look like {'eval': {'0': {...}, ...}}. "
            f"Got keys={list(data.keys())[:10]}"
        )
    return data["eval"]

def _load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def _extract_single_variant_payload(
    json_path: Path,
    *,
    prompt_type_default: str,
    sample_key_default: str,
    variant_index_default: int,
) -> Tuple[str, str, int, str, Dict[str, Any]]:
    """
    Accept either:
      1) eval-style JSON: {"eval": {"0": {... "prompt_B_dict_refined": {"diversity_variants": [...]}}}}
      2) minimal single-variant JSON:
         {
           "prompt_type": "B",
           "sample_key": "0",
           "variant_index": 0,
           "general_prompt": "...",        # optional
           "variant": { ... }              # REQUIRED
         }
      3) raw variant dict JSON: { ... } (treated as "variant" directly)
    Returns: (prompt_type, sample_key, variant_index, general_prompt, variant_dict)
    """
    data = _load_json(json_path)
    prompt_type = str(prompt_type_default).upper()
    sample_key = str(sample_key_default)
    variant_index = int(variant_index_default)
    general_prompt = ""

    if isinstance(data, dict) and "eval" in data:
        eval_entries = data.get("eval")
        if not isinstance(eval_entries, dict):
            raise ValueError("Expected `eval` to be a dict in eval-style JSON.")
        entry = eval_entries.get(sample_key)
        if entry is None:
            # Fall back to first key if requested one is missing.
            sample_key = next(iter(eval_entries.keys()))
            entry = eval_entries[sample_key]
        if not isinstance(entry, dict):
            raise ValueError(f"Sample entry for key {sample_key} is not a dict.")
        prompt_block = _get_prompt_block(entry, prompt_type)
        variants = prompt_block.get("diversity_variants", [])
        if not isinstance(variants, list) or not variants:
            raise ValueError(f"prompt_{prompt_type}_dict_refined has no `diversity_variants` for sample {sample_key}.")
        if variant_index < 0 or variant_index >= len(variants):
            raise IndexError(f"variant_index={variant_index} out of range (0..{len(variants)-1}).")
        variant = variants[variant_index]
        if not isinstance(variant, dict):
            raise ValueError(f"Variant at index {variant_index} is not a dict.")
        general_prompt = _get_general_prompt(entry)
        return prompt_type, sample_key, variant_index, general_prompt, variant

    # Minimal single-variant payload
    if isinstance(data, dict) and "variant" in data:
        variant = data.get("variant")
        if not isinstance(variant, dict):
            raise ValueError("In minimal JSON, `variant` must be a dict.")
        prompt_type = str(data.get("prompt_type", prompt_type)).upper()
        sample_key = str(data.get("sample_key", sample_key))
        variant_index = int(data.get("variant_index", variant_index))
        general_prompt = str(data.get("general_prompt", general_prompt))
        return prompt_type, sample_key, variant_index, general_prompt, variant

    # Raw variant dict
    if isinstance(data, dict):
        variant = data
        return prompt_type, sample_key, variant_index, general_prompt, variant

    raise ValueError(f"Unsupported JSON format in {json_path} (type={type(data)}).")


def _get_prompt_block(sample_entry: Dict[str, Any], prompt_type: str) -> Dict[str, Any]:
    prompt_type = prompt_type.upper()
    refined_key = f"prompt_{prompt_type}_dict_refined"
    base_key = f"prompt_{prompt_type}_dict"
    block = sample_entry.get(refined_key) or sample_entry.get(base_key)
    if not isinstance(block, dict):
        raise KeyError(f"Missing prompt block {refined_key} (or {base_key}).")
    return block


def _get_general_prompt(sample_entry: Dict[str, Any]) -> str:
    block = sample_entry.get("prompt_A_dict_refined") or sample_entry.get("prompt_A_dict")
    if isinstance(block, dict):
        value = block.get("ori_prompt")
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = sample_entry.get("general_prompt")
    if isinstance(value, str):
        return value.strip()
    return ""


def _variant_to_runtime(
    variant: Dict[str, Any],
    *,
    fallback_resolution: int,
) -> Tuple[List[Any], List[Any], int, int, Optional[List[List[int]]]]:
    prompts = variant.get("detailed_caption_list") or []
    rewrite_prompts = variant.get("rewrite_detailed_caption_list") or []
    if not isinstance(prompts, list) or len(prompts) == 0:
        raise ValueError("Variant missing non-empty `detailed_caption_list`.")
    if not isinstance(rewrite_prompts, list):
        rewrite_prompts = []

    W = int(variant.get("width") or fallback_resolution)
    H = int(variant.get("height") or fallback_resolution)

    layout_plan = variant.get("layout_plan")
    if isinstance(layout_plan, list) and layout_plan and W > 0 and H > 0:
        bboxes = [
            [int(bb[0] * W), int(bb[1] * H), int(bb[2] * W), int(bb[3] * H)]
            for bb in layout_plan
        ]
    else:
        bboxes = None

    return prompts, rewrite_prompts, W, H, bboxes


_DEFAULT_CHECKPOINT_REPO_ID = "lilyecho/people-composer"
_CHECKPOINT_EXTENSIONS = (".pt", ".pth", ".bin", ".safetensors")
_CHECKPOINT_PRIORITY_NAMES = (
    "people_composer.pt",
    "people-composer.pt",
    "model.pt",
    "checkpoint.pt",
    "pytorch_model.bin",
    "model.safetensors",
)


def _load_checkpoint_into_model(model, ckpt_path: Path) -> None:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(ckpt_path), device="cpu")
    else:
        state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _select_checkpoint_filename(repo_id: str) -> str:
    files = HfApi().list_repo_files(repo_id=repo_id, repo_type="model")
    checkpoint_files = [
        name
        for name in files
        if name.lower().endswith(_CHECKPOINT_EXTENSIONS) and not name.endswith("/")
    ]
    if not checkpoint_files:
        raise FileNotFoundError(
            f"No checkpoint file with extension {', '.join(_CHECKPOINT_EXTENSIONS)} "
            f"was found in model repo `{repo_id}`."
        )

    by_basename = {Path(name).name: name for name in checkpoint_files}
    for preferred in _CHECKPOINT_PRIORITY_NAMES:
        if preferred in by_basename:
            return by_basename[preferred]

    if len(checkpoint_files) == 1:
        return checkpoint_files[0]

    candidates = ", ".join(checkpoint_files)
    raise ValueError(
        "Multiple checkpoint files were found. Set `checkpoint_filename` in "
        f"`configs/inference.yaml` to one of: {candidates}"
    )


def _resolve_checkpoint_path(args: Any) -> Path:
    ckpt_path_value = getattr(args, "checkpoint_path", None)
    if _is_nonempty_string(ckpt_path_value):
        ckpt_path = Path(str(ckpt_path_value)).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    repo_id = str(getattr(args, "checkpoint_repo_id", _DEFAULT_CHECKPOINT_REPO_ID) or _DEFAULT_CHECKPOINT_REPO_ID)
    filename_value = getattr(args, "checkpoint_filename", "")
    try:
        filename = str(filename_value).strip() if _is_nonempty_string(filename_value) else _select_checkpoint_filename(repo_id)
        cache_dir = Path(str(getattr(args, "checkpoint_cache_dir", "checkpoints/people-composer"))).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
        )
    except (EntryNotFoundError, RepositoryNotFoundError, HfHubHTTPError) as exc:
        raise RuntimeError(
            "Could not download the PeopleComposer checkpoint. If the model repo "
            "is private or not public yet, make sure you are logged in with "
            "`huggingface-cli login` or set `checkpoint_path` to a local file."
        ) from exc
    return Path(downloaded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-GPU, single-sample inference (refine)")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config.")
    parser.add_argument(
        "--sample-json",
        type=str,
        default=None,
        help="Override the eval JSON path (expects top-level key `eval`).",
    )
    parser.add_argument("--sample-key", type=str, default=None, help="Which sample key in the eval JSON to run.")
    parser.add_argument("--type", dest="prompt_type", choices=["A", "B", "C"], default=None)
    parser.add_argument("--variant-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()

def draw_bboxes_on_existing_image(
    image: Image.Image,
    bboxes: list[list[int]] = None,
    bbox_colors: list[str] = None,
    line_width: int = 3,
    ) -> Image.Image:
    """
    Draw bounding boxes on an existing PIL Image
    """
    if bboxes is None:
        return image
    if bbox_colors is None:
        bbox_colors = ["red", "green", "blue", "purple", "orange"]

    # Create a copy to avoid modifying the original
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)

    # Draw each bounding box on the image
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        color = bbox_colors[i % len(bbox_colors)]  # Cycle through colors
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

    return img_with_boxes

def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as tmp_file:
            json.dump(payload, tmp_file, indent=2)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        tmp_path.replace(path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except TypeError:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass


def sample_step(
    dit,
    args,
    prompt_idx,
    prompt_dict,
    var_idx,
    prompts,
    rewrite_prompts,
    general_prompt,
    bounding_boxes_in_order,
    W,
    H,
    global_step,
    wandbrun,
    *,
    variant_tag: str,
    run_type: str,
    offload=True,
    seed_idx=0,
    save_dir=None,
    seed=182396138,
    html_output_path=None,
    target_dir_override=None,
    filename_stub_override=None,
    final_round_filename=None,
    visualization_filename=None,
):
    # Use provided seed or fallback to a random one
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    print(
            f"🔍 DEBUG: Inside sample_step - received idx={prompt_idx}, var_idx={var_idx}, seed={seed}, seed_idx={seed_idx}"
        )

    image_name = (
        f"Inference Results for step {global_step}, prompt {prompt_idx}, variant {var_idx}, tag {variant_tag}, {run_type}, seed {seed}"
    )
    local_gpu = torch.cuda.current_device()
    print(
        f"🎨 Generating images: step={global_step}, prompt_idx={prompt_idx}, variant={var_idx}, tag={variant_tag}, run_type={run_type}, seed={seed}"
    )
    sampler = XFluxSampler(clip=None, t5=None, ae=None, model=dit, device=f"cuda:{local_gpu}", offload=offload)


    if target_dir_override is not None:
        target_dir = os.fspath(target_dir_override)
    else:
        target_dir = os.path.join(
            save_dir,
            args.inference_output_dir,
            f"prompt_{prompt_idx}",
        )
    os.makedirs(target_dir, exist_ok=True)

    filename_stub = filename_stub_override or f"var_{var_idx}_tag_{variant_tag}_seed_{seed}_{run_type}"
    primary_real_image = os.path.join(target_dir, f"{filename_stub}_refine_r1_real.jpg")
    legacy_primary_image = os.path.join(target_dir, f"var_{var_idx}_seed_{seed}_refine_r1_real.jpg")
    if os.path.exists(primary_real_image) or os.path.exists(legacy_primary_image):
        print(
            f"↩️ Skipping prompt {prompt_idx}, variant {var_idx}, tag {variant_tag}, run {run_type}, seed {seed}: outputs already exist."
        )
        return False

    # Use iterative sampling with multiple rounds
    rounds_output, clip, t5, vae = sampler.forward_multiperson_morerounds_rewrite(
        prompts=prompts,
        rewrite_prompts=rewrite_prompts,
        general_prompt=general_prompt,
        width=W,
        height=H,
        num_steps=args.sample_steps,
        seed=seed,
        customize_img_ids=args.customize_img_ids,
        customize_txt_ids=args.customize_txt_ids,
        bounding_boxes_in_order=bounding_boxes_in_order,
        use_v1_bbox=args.use_v1_bbox,
        guidance=args.guidance,
        attn_mask_type=args.get("attn_mask_type", None),
    )

    # Helper to create a centered text banner of given width
    def _create_text_banner(text: str, width: int, font: ImageFont.FreeTypeFont, padding: int = 10, bg_color: str = "white", text_color: str = "black"):
        draw_dummy = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        if hasattr(draw_dummy, "textbbox"):
            bbox_default = draw_dummy.textbbox((0, 0), "Ag", font=font)
            default_line_h = bbox_default[3] - bbox_default[1]
        else:
            try:
                _, default_line_h = draw_dummy.textsize("Ag", font=font)
            except AttributeError:
                bbox_default = font.getbbox("Ag")
                default_line_h = bbox_default[3] - bbox_default[1]
        max_text_width = width - 2 * padding
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            if hasattr(draw_dummy, "textbbox"):
                bbox = draw_dummy.textbbox((0, 0), test_line, font=font)
                line_w = bbox[2] - bbox[0]
            else:
                try:
                    line_w, _ = draw_dummy.textsize(test_line, font=font)
                except AttributeError:
                    bbox = font.getbbox(test_line)
                    line_w = bbox[2] - bbox[0]
            if line_w <= max_text_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        if not lines:
            lines = [""]
        text_height = default_line_h
        banner_height = len(lines) * text_height + (len(lines) + 1) * padding
        banner = Image.new('RGB', (width, banner_height), color=bg_color)
        draw = ImageDraw.Draw(banner)
        y = padding
        for line in lines:
            if hasattr(draw_dummy, "textbbox"):
                bbox = draw_dummy.textbbox((0, 0), line, font=font)
                text_w = bbox[2] - bbox[0]
            else:
                try:
                    text_w, _ = draw_dummy.textsize(line, font=font)
                except AttributeError:
                    bbox = font.getbbox(line)
                    text_w = bbox[2] - bbox[0]
            draw.text(((width - text_w) // 2, y), line, fill=text_color, font=font)
            y += text_height + padding
        return banner

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=16)
    except Exception:
        font = ImageFont.load_default()

    round_keys = sorted(rounds_output.keys(), key=lambda x: int(x))
    per_round_images = []

    num_rounds = len(round_keys)

    for round_idx, key in enumerate(round_keys):
        round_data = rounds_output[key]
        pose_img = round_data["pose_img"]
        real_img = round_data["real_img"]
        segmented_img = round_data.get("segmented_image", None)

        if pose_img.mode != 'RGB':
            pose_img = pose_img.convert('RGB')
        if real_img.mode != 'RGB':
            real_img = real_img.convert('RGB')
        if segmented_img is not None and segmented_img.mode != 'RGB':
            segmented_img = segmented_img.convert('RGB')

        # Save pure pose image (without bounding boxes) first
        r_label = round_idx + 1
        pose_img_with_bbox = pose_img  # Keep reference to original for bbox version

        try:
            # Save pure pose image
            pose_img.save(os.path.join(target_dir, f"{filename_stub}_refine_r{r_label}_pose_pure.jpg"), format="JPEG", quality=95)
            real_img.save(os.path.join(target_dir, f"{filename_stub}_refine_r{r_label}_real.jpg"), format="JPEG", quality=95)
            if segmented_img is not None:
                segmented_img.save(os.path.join(target_dir, f"{filename_stub}_refine_r{r_label}_seg.jpg"), format="JPEG", quality=95)
        except Exception as e:
            print(f"❌ Failed to save per-round images for prompt {prompt_idx}, var {var_idx}, round {r_label}, seed {seed}: {e}")
        if final_round_filename and (round_idx == num_rounds - 1):
            final_real_path = os.path.join(target_dir, final_round_filename)
            try:
                real_img.save(final_real_path, format="JPEG", quality=95)
            except Exception as exc:
                print(f"⚠️ Failed to save final real image copy {final_real_path}: {exc}")

        # Now add bounding boxes for visualization and save bbox version
        if bounding_boxes_in_order is not None:
            bboxes_to_show = bounding_boxes_in_order[:round_idx+1]
            from_w, from_h = W, H
            to_w, to_h = pose_img.width, pose_img.height
            if (from_w != to_w) or (from_h != to_h):
                scale_x = to_w / from_w
                scale_y = to_h / from_h
                bboxes_scaled = [
                    [
                        int(b[0] * scale_x),
                        int(b[1] * scale_y),
                        int(b[2] * scale_x),
                        int(b[3] * scale_y),
                    ]
                    for b in bboxes_to_show
                ]
            else:
                bboxes_scaled = bboxes_to_show
            print(f"🎯 Round {key}: Drawing {len(bboxes_scaled)} bounding boxes on pose image at size {pose_img.width}x{pose_img.height}")
            pose_img_with_bbox = draw_bboxes_on_existing_image(pose_img, bboxes_scaled, line_width=2)

            # Save pose image with bounding boxes
            try:
                pose_img_with_bbox.save(os.path.join(target_dir, f"{filename_stub}_refine_r{r_label}_pose.jpg"), format="JPEG", quality=95)
            except Exception as e:
                print(f"❌ Failed to save pose image with bbox for prompt {prompt_idx}, var {var_idx}, round {r_label}, seed {seed}: {e}")
        else:
            # If no bounding boxes, the pose image with bbox is the same as pure pose
            pose_img_with_bbox = pose_img

        images_to_concat = [pose_img_with_bbox, real_img]
        if segmented_img is not None:
            images_to_concat.append(segmented_img)
            print(f"🖼️ Round {key}: Including segmented image in visualization")

        concat_width = sum(img.width for img in images_to_concat)
        concat_height = max(img.height for img in images_to_concat)
        concat_img = Image.new('RGB', (concat_width, concat_height), color='white')

        x_offset = 0
        for img in images_to_concat:
            concat_img.paste(img, (x_offset, 0))
            x_offset += img.width

        title_banner = _create_text_banner(f"Round {key}", concat_width, font)
        prompt_text = rewrite_prompts[round_idx]['caption'] if round_idx < len(prompts) else ""
        prompt_banner = _create_text_banner(prompt_text, concat_width, font)

        total_h = title_banner.height + concat_img.height + prompt_banner.height
        round_img = Image.new('RGB', (concat_width, total_h), color='white')
        y_offset = 0
        round_img.paste(title_banner, (0, y_offset)); y_offset += title_banner.height
        round_img.paste(concat_img, (0, y_offset)); y_offset += concat_img.height
        round_img.paste(prompt_banner, (0, y_offset))

        per_round_images.append(round_img)

    final_width = max(img.width for img in per_round_images)
    general_banner = _create_text_banner(general_prompt, final_width, font)
    seed_banner = _create_text_banner(
        f"Seed: {seed} | Tag: {variant_tag} | Run: {run_type}",
        final_width,
        font,
        bg_color="lightblue",
    )
    final_height = general_banner.height + seed_banner.height + sum(img.height for img in per_round_images)

    final_img = Image.new('RGB', (final_width, final_height), color='white')
    y_offset = 0
    final_img.paste(general_banner, (0, y_offset)); y_offset += general_banner.height
    final_img.paste(seed_banner, (0, y_offset)); y_offset += seed_banner.height
    for img in per_round_images:
        final_img.paste(img, (0, y_offset))
        y_offset += img.height




    print(f"🔍 DEBUG: About to save with idx={prompt_idx}, var_idx={var_idx}, seed={seed}")
    final_filename = visualization_filename or f"{filename_stub}_refine.jpg"
    save_path = os.path.join(target_dir, final_filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        final_img.save(save_path, format="JPEG", quality=95)
        print(f"💾 Saved: {save_path}")
    except Exception as e:
        print(f"❌ Failed to save {save_path}: {e}")
        return False

    seed_input_payload = {
        "prompt_idx": prompt_idx,
        "variant_idx": var_idx,
        "variant_tag": variant_tag,
        "run_type": run_type,
        "seed": seed,
        "general_prompt": general_prompt,
        "prompts": prompts,
        "width": W,
        "height": H,
        "bounding_boxes_in_order": bounding_boxes_in_order,
        "prompt_dict": prompt_dict,
    }
    seed_input_path = os.path.join(target_dir, f"{filename_stub}_input.json")
    try:
        _atomic_write_json(Path(seed_input_path), seed_input_payload)
    except Exception as e:
        print(f"⚠️ Failed to save seed input JSON for prompt {prompt_idx}, var {var_idx}, seed {seed}: {e}")
    # if html_output_path is not None:
    #     _refresh_html_gallery(
    #         base_save_dir=save_dir,
    #         inference_dir=args.inference_output_dir,
    #         json_path=args.sample_prompts_json,
    #         seeds=[],  # scan directory for any seeds
    #         html_filename=html_output_path,
    #     )

    return True

def main() -> None:
    cli = parse_args()
    args = OmegaConf.load(cli.config)

    sample_json = (
        cli.sample_json
        or getattr(args, "single_variant_json", None)
        or getattr(args, "single_sample_json", None)
        or getattr(args, "sample_prompts_json", None)
    )
    if not sample_json:
        raise ValueError("Provide --sample-json or set `single_sample_json` in the YAML.")
    sample_json_path = Path(sample_json)

    prompt_type = (cli.prompt_type or getattr(args, "prompt_type", "A")).upper()
    variant_index = int(cli.variant_index if cli.variant_index is not None else getattr(args, "variant_index", 0))
    seed = int(cli.seed if cli.seed is not None else getattr(args, "seed", 182396138))

    output_dir = cli.output_dir or getattr(args, "output_dir", "single_sample_outputs")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Single GPU only (no workers, no queue). Use whatever GPU is visible.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this inference script, but torch.cuda.is_available() is False.")

    sample_key = str(cli.sample_key or getattr(args, "sample_key", "0"))
    prompt_type, sample_key, variant_index, general_prompt, variant = _extract_single_variant_payload(
        sample_json_path,
        prompt_type_default=prompt_type,
        sample_key_default=sample_key,
        variant_index_default=variant_index,
    )

    resolution = int(getattr(args, "resolution", 1024))
    prompts, rewrite_prompts, W, H, bboxes = _variant_to_runtime(variant, fallback_resolution=resolution)

    # Model setup (copied from the main eval script, minus multi-GPU and queues).
    local_gpu = torch.cuda.current_device()
    dit = load_flow_model2(getattr(args, "model_name", "flux-dev"), device="cpu")
    if bool(getattr(args, "use_lora", True)):
        replace_attn_processor_triplelora_ar(dit, args)
    replace_split_head(dit, args)
    dit.eval()
    dit.to(torch.bfloat16)

    ckpt_path = _resolve_checkpoint_path(args)
    _load_checkpoint_into_model(dit, ckpt_path)

    # Output structure: keep multi-type-style layout, but only one variant.
    run_dir = output_root / f"prompt_{prompt_type}" / f"sample_{sample_key}" / f"variant_{variant_index}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save a tiny provenance bundle.
    OmegaConf.save(config=args, f=str(output_root / "config_single_sample.yaml"))
    (output_root / "sample_json_path.txt").write_text(str(sample_json_path) + "\n")
    (output_root / "checkpoint_path.txt").write_text(str(ckpt_path) + "\n")

    # Ensure sampler args are consistent with resolution.
    args.sample_width = W
    args.sample_height = H
    args.training_width = W
    args.training_height = H
    args.img_seq_len = (W // 16) * (H // 16)
    args.cond_seq_len = (W // 16) * (H // 16)

    args.guidance = float(getattr(args, "guidance", 4.0))
    args.use_v1_bbox = bool(getattr(args, "use_v1_bbox", False))

    ok = sample_step(
        dit,
        args,
        int(sample_key) if sample_key.isdigit() else 0,
        variant,
        variant_index,
        prompts,
        rewrite_prompts,
        general_prompt,
        bboxes,
        W,
        H,
        global_step=0,
        wandbrun=None,
        variant_tag=f"{prompt_type}_variant_{variant_index}",
        run_type=f"prompt_{prompt_type}",
        offload=bool(getattr(args, "offload_when_sample", True)),
        seed_idx=0,
        save_dir=str(output_root),
        seed=seed,
        html_output_path=None,
        target_dir_override=str(run_dir),
        filename_stub_override=f"seed_{seed}",
        final_round_filename=f"seed_{seed}_final.jpg",
        visualization_filename=f"seed_{seed}_visualization.jpg",
    )

    print(f"✅ Done (generated={ok}). Output dir: {run_dir} (GPU={local_gpu})")


if __name__ == "__main__":
    main()

import torch
from einops import rearrange
from torch import Tensor
from typing import Optional

def infer_W_from_wrapped_w(col_w_prime: torch.Tensor) -> int:
    # col_w_prime: [N] float, transformed w' along row-major flattened grid
    w = col_w_prime.detach()
    diff = w[1:] - w[:-1]
    wrap = (diff <= 0)  # detect the big drop at row boundary
    idx = wrap.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        raise ValueError("Could not detect row wrap; is this truly row-major flattened?")
    return int(idx[0].item() + 1)


def adjust_condition_ids_inplace(
    batch_ids: torch.Tensor,                      # [B, S, 3], dtype float (so we can write fractional coords)
    batch_multi_person_last_indicator,            # list[bool] or BoolTensor of len B
    batch_shift: torch.Tensor,                    # [B, 2]  -> (x, y) per sample
    batch_resize_scale: torch.Tensor,             # [B]     -> scalar per sample
    ori_h: int,
    ori_w: int,
    txt_len: int = 512,
):
    """
    Modifies `batch_ids` IN-PLACE for img1 segment only, for samples where indicator is True.

    For each affected sample b:
      - Let L = (S - txt_len) / 3; img1 segment is [txt_len : txt_len + L]
      - Re-inflate img1 (originally flattened H*W x 3) back to [H, W, 3]
      - Update:
            h := ((h + 0.5) / resize_scale_b) + y_b
            w := ((w + 0.5) / resize_scale_b) + x_b
      - Writes results back in-place (keeps autograd chain for h,w)

    Assumptions:
      - batch_ids dtype is floating (so fractional results are kept).
      - The 3rd dim of ids is [0, h_i, w_i].
      - img1 was produced by flattening a HxW grid (row-major), so we can recover W via max(w_i)+1.
    """
    # print("adjust_condition_ids_inplace")
    # print("batch_ids", batch_ids)
    if not torch.is_tensor(batch_ids):
        raise TypeError("batch_ids must be a torch.Tensor")
    if batch_ids.ndim != 3 or batch_ids.size(-1) != 3:
        raise ValueError(f"batch_ids must have shape [B, S, 3], got {tuple(batch_ids.shape)}")
    if not batch_ids.dtype.is_floating_point:
        raise TypeError("batch_ids must be a floating dtype to store fractional coordinates.")

    B, S, C = batch_ids.shape # B always 1 now
    if (S - txt_len) % 3 != 0:
        raise ValueError(f"S - txt_len must be divisible by 3; got S={S}, txt_len={txt_len}")
    L = (S - txt_len) // 3
    img1_start, img1_end = txt_len, txt_len + L

    # Normalize indicator -> Python bool list for simple looping
    if batch_multi_person_last_indicator.shape[0] != B:
        raise ValueError("indicator length must equal batch size")
    if batch_shift.shape != (B, 2):
        raise ValueError(f"batch_shift must be [B,2], got {tuple(batch_shift.shape)}")
    if batch_resize_scale.shape != (B,):
        raise ValueError(f"batch_resize_scale must be [B], got {tuple(batch_resize_scale.shape)}")

    new_batch_ids = []
    for b in range(B):
        if not bool(batch_multi_person_last_indicator[b]):
            new_batch_ids.append(batch_ids[b])
            continue

        # img1 flattened view: [L, 3] with columns [0, h, w]
        img1_flat = batch_ids[b, img1_start:img1_end, :]  # view into batch_ids (no copy) -- [L, 3]
        #print("img1_flat", img1_flat)
        # Recover W from max w-index; this touches only index-like data, safe to detach.
        # Inflate to [H, W, 3] as a view; modifying this will modify batch_ids in-place.
        # .view requires contiguous; your flat path was view(-1,3), so this should be contiguous too.
        # If not, you can replace with .reshape(H, W, 3).
        img1_grid = img1_flat.view(ori_h[b], ori_w[b], 3)

        xs = torch.arange(ori_w[b], dtype=torch.float32) + 0.5   # centers along x
        ys = torch.arange(ori_h[b], dtype=torch.float32) + 0.5  # centers along y
        # Meshgrid with image-style indexing: i=x (cols), j=y (rows). 
        # We'll produce arrays in shape (H, W) indexed dy[j,i], dx[j,i].
        # Meshgrid with image-style indexing: i=x (cols), j=y (rows).
        Xc, Yc = torch.meshgrid(xs.to(batch_ids.device), ys.to(batch_ids.device), indexing="xy")
        xb, yb = batch_shift[b]
        
        cell_x = xb + Xc / batch_resize_scale[b]
        cell_y = yb + Yc / batch_resize_scale[b]
        
        centers = torch.stack([cell_y, cell_x], dim=-1)

        # Create a zero channel
        zero_channel = torch.full((ori_h[b], ori_w[b], 1), 1, dtype=centers.dtype, device=centers.device)

        # Concatenate along the last dimension
        img_ids_updated = torch.cat([zero_channel, centers], dim=-1)
        img_ids_updated_flat = img_ids_updated.view(-1, 3)
        this_batch_ids = torch.cat([batch_ids[b, :img1_start, :], img_ids_updated_flat, batch_ids[b, img1_end:, :]], dim=0)
        new_batch_ids.append(this_batch_ids)

    return torch.stack(new_batch_ids, dim=0)


def adjust_condition_ids_inplace_v2(
    batch_ids: torch.Tensor,                      # [B, S, 3], dtype float (so we can write fractional coords)
    batch_multi_person_last_indicator,            # list[bool] or BoolTensor of len B
    batch_shift: torch.Tensor,                    # [B, 2]  -> (x, y) per sample
    batch_resize_scale: torch.Tensor,             # [B]     -> scalar per sample
    ori_h: int,
    ori_w: int,
    txt_len: int = 512,
):
    """
    Modifies `batch_ids` IN-PLACE for img1 segment only, for samples where indicator is True.

    For each affected sample b:
      - Let L = (S - txt_len) / 3; img1 segment is [txt_len : txt_len + L]
      - Re-inflate img1 (originally flattened H*W x 3) back to [H, W, 3]
      - Update:
            h := ((h + 0.5) / resize_scale_b) + y_b
            w := ((w + 0.5) / resize_scale_b) + x_b
      - Writes results back in-place (keeps autograd chain for h,w)

    Assumptions:
      - batch_ids dtype is floating (so fractional results are kept).
      - The 3rd dim of ids is [0, h_i, w_i].
      - img1 was produced by flattening a HxW grid (row-major), so we can recover W via max(w_i)+1.
    """
    # print("adjust_condition_ids_inplace")
    # print("batch_ids", batch_ids)
    if not torch.is_tensor(batch_ids):
        raise TypeError("batch_ids must be a torch.Tensor")
    if batch_ids.ndim != 3 or batch_ids.size(-1) != 3:
        raise ValueError(f"batch_ids must have shape [B, S, 3], got {tuple(batch_ids.shape)}")
    if not batch_ids.dtype.is_floating_point:
        raise TypeError("batch_ids must be a floating dtype to store fractional coordinates.")

    B, S, C = batch_ids.shape # B always 1 now
    if (S - txt_len) % 3 != 0:
        raise ValueError(f"S - txt_len must be divisible by 3; got S={S}, txt_len={txt_len}")
    L = (S - txt_len) // 3
    img1_start, img1_end = txt_len, txt_len + L

    # Normalize indicator -> Python bool list for simple looping
    if batch_multi_person_last_indicator.shape[0] != B:
        raise ValueError("indicator length must equal batch size")
    if batch_shift.shape != (B, 2):
        raise ValueError(f"batch_shift must be [B,2], got {tuple(batch_shift.shape)}")
    if batch_resize_scale.shape != (B,):
        raise ValueError(f"batch_resize_scale must be [B], got {tuple(batch_resize_scale.shape)}")


    for b in range(B):
        if not bool(batch_multi_person_last_indicator[b]):
            continue

        # img1 flattened view: [L, 3] with columns [0, h, w]
        img1_flat = batch_ids[b, img1_start:img1_end, :]  # view into batch_ids (no copy) -- [L, 3]
        #print("img1_flat", img1_flat)
        # Recover W from max w-index; this touches only index-like data, safe to detach.
        # Inflate to [H, W, 3] as a view; modifying this will modify batch_ids in-place.
        # .view requires contiguous; your flat path was view(-1,3), so this should be contiguous too.
        # If not, you can replace with .reshape(H, W, 3).
        img1_grid = img1_flat.view(ori_h[b], ori_w[b], 3)

        # Grab h,w planes (views). We'll compute out-of-place, then copy_ back to avoid versioning issues.
        h_plane = img1_grid[..., 1]   # [H, W]
        w_plane = img1_grid[..., 2]   # [H, W]

        # Per-sample params as rank-0 tensors on the right device (for broadcasting)
        x_b, y_b = batch_shift[b]           # shape [], []
        rs_b = batch_resize_scale[b]        # shape []

        # Compute new values (no .item() on h/w; keep everything differentiable)
        new_h = (h_plane + 0.5) / rs_b + y_b
        new_w = (w_plane + 0.5) / rs_b + x_b

        # In-place write-back using copy_ (safer for autograd than chained in-place arithmetic)
        h_plane.copy_(new_h)
        w_plane.copy_(new_w)

    return batch_ids



def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, attn_mask: Optional[Tensor] = None, bbox_id_embeddings: Optional[Tensor] = None) -> Tensor:
    q, k = apply_rope(q, k, pe)

    if bbox_id_embeddings is not None:
        num_text_tokens = 512
        q_img_tokens = q[:, :, num_text_tokens:]
        k_img_tokens = k[:, :, num_text_tokens:]
        
        d_embed = bbox_id_embeddings.shape[-1] # 16
        broadcastable_bbox_embeddings = bbox_id_embeddings.unsqueeze(1)
        
        # CORRECT SLICE: The missing colon tells PyTorch to select all of the L_img dimension
        # The slice is now [B, H, L_img, 16]
        q_img_tokens[:, :, :, :d_embed] += broadcastable_bbox_embeddings
        k_img_tokens[:, :, :, :d_embed] += broadcastable_bbox_embeddings

    if attn_mask is not None:
        # Unsqueeze (add a new dimension of size 1) at position 1 (the head dimension)
        assert attn_mask.ndim == 3, "attn_mask must be [B, L, L]"
        attn_mask = attn_mask.unsqueeze(1) # Now shape (B, 1, L, L)
        # This (B, 1, L, L) mask will then be broadcasted to (B, H, L, L)
        # by scaled_dot_product_attention because H is 24 (from your error message).


    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x



def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()

def rope_learnable(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()

def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    # print("DEBUG -----------------------------------")
    # print("xq.shape!!", xq.shape)
    # print("xk.shape!!", xk.shape)
    # print("freqs_cis.shape!!", freqs_cis.shape)
    # print("DEBUG -----------------------------------")
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

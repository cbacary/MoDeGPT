# Type 3 compression

import logging

import torch
from torch.types import Tensor

from compression_utils import get_V_O_weights, slice_VO_dims, sqrt_M
from model_utils import get_model_attrs, dtype_p, d1, d2

logger = logging.getLogger("MoDeGPT")


@torch.no_grad()
def compress_vo(
    model,
    cov: list[Tensor],
    keep_ratios=None,
    ridge_lambda=1e-4,
    slice_dims=True,
):
    n_layers, n_heads, d_model, head_dim, arch = get_model_attrs(model)
    for layer in range(n_layers):
        keep_ratio = keep_ratios[layer]
        rank_i = int(head_dim * keep_ratio)
        rank_i = max(1, rank_i)

        C = cov[layer].to(device=d2)
        sqrt_C = sqrt_M(C)
        inv_sqrt_C = torch.linalg.inv(sqrt_C)

        try:
            W_v, W_o = get_V_O_weights(model=model, layer_idx=layer)
        except Exception as e:
            logger.warning(f"[VO] Layer {layer}: cannot access v_proj/o_proj: {e}")
            continue

        new_heads_V = []
        new_heads_O = []

        for h in range(n_heads):
            compress_head(
                head_idx=h,
                head_dim=head_dim,
                rank_i=rank_i,
                layer=layer,
                W_v=W_v,
                W_o=W_o,
                sqrt_C=sqrt_C,
                inv_sqrt_C=inv_sqrt_C,
                new_heads_V=new_heads_V,
                new_heads_O=new_heads_O,
                slice_dims=slice_dims,
            )

        if slice_dims:
            slice_VO_dims(
                model=model,
                layer_idx=layer,
                new_heads_V=new_heads_V,
                new_heads_O=new_heads_O,
                bias=True,
            )

        if logger:
            logger.info(
                f"[VO] ✅ Compressed layer {layer} to rank {rank_i} per head (λ={ridge_lambda})"
            )
        torch.cuda.empty_cache()


def compress_head(
    head_idx: int,
    head_dim: int,
    rank_i: int,
    layer: int,
    W_v: Tensor,
    W_o: Tensor,
    sqrt_C: Tensor,
    inv_sqrt_C: Tensor,
    new_heads_V: list[Tensor],
    new_heads_O: list[Tensor],
    slice_dims=True,
) -> tuple[Tensor, Tensor]:
    # head_start_idx, head_end_idx
    head_s, head_e = head_idx * head_dim, (head_idx + 1) * head_dim
    # head_dims = head_dims, d_model = hidden dimensions
    # V_head [head_dims, d_model], O_head [d_model, head_dims]
    # sqrt_C [d_model, d_model]

    V_head = W_v[head_s:head_e, :].to(dtype=dtype_p, device=d2)
    O_head = W_o[:, head_s:head_e].to(dtype=dtype_p, device=d2)

    # V_head needs to be transposed because we expect a tensor of (in_features, out_features)
    # but torch stores weights as (out_features, in_features)
    U, _S, V = torch.linalg.svd(sqrt_C @ V_head.T, full_matrices=False)

    # sqrt_C @ V_head.T [d_model, head_dims]
    # U [d_model, head_dims], S [head_dims, head_dims], V [head_dims, head_dims]
    S = torch.diag(_S)  # [head_dims, head_dims]

    A = S @ V @ O_head.T  # once again transpose O_head
    # There is marginal difference (from minimal testing) between full_matrices=False|True
    U_p, _S_p, V_p = torch.linalg.svd(A, full_matrices=True)

    S_p = torch.diag(_S_p)

    # A [head_dims, d_model]
    # U_p [head_dims, head_dims], S_p [d_model, d_model], V_p [d_model, d_model]

    # [d_model, d_model] @ [d_model, head_dims] @ [head_dims, d_model] =
    # = [d_model, head_dims] @ [head_dims, head_dims] =
    # = [d_model, head_dims]
    compressed_v = (inv_sqrt_C @ U @ U_p)[:, :rank_i]
    # [d_model, d_model] @ [d_model, d_model]
    compressed_o = S_p[:rank_i, :rank_i] @ V_p[:rank_i, :]

    # compressed_v = compressed_v.T
    # compressed_o = compressed_o.T

    if slice_dims:
        new_heads_V.append(compressed_v.T)
        new_heads_O.append(compressed_o.T)
    else:
        V_new = torch.zeros_like(V_head).to(dtype=W_v.dtype, device=W_v.device)
        O_new = torch.zeros_like(O_head).to(dtype=W_o.dtype, device=W_o.device)

        # Retranspose the output because we transposed the input
        V_new[:rank_i, :] = compressed_v.T
        O_new[:, :rank_i] = compressed_o.T

        W_v[head_s:head_e, :].data.copy_(V_new)
        W_o[:, head_s:head_e].data.copy_(O_new)

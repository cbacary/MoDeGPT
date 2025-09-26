import logging

import torch
from torch.types import Tensor

from compression_utils import sqrt_M

logger = logging.getLogger("MoDeGPT")


@torch.no_grad()
def compress_vo(
    model,
    cov: list[Tensor],
    keep_ratios=None,
    rank=None,
    n_layers=None,
    n_heads=None,
    head_dim=None,
    ridge_lambda=1e-4,
    min_rank=16,
    max_condition_number=1e4,
):
    for layer in range(n_layers):
        keep_ratio = keep_ratios[layer]
        rank_i = int(head_dim * keep_ratio) if rank is None else rank
        rank_i = max(min_rank, min(rank_i, head_dim))

        C = cov[layer].to(device="cuda")
        sqrt_C = sqrt_M(C)
        inv_sqrt_C = torch.linalg.inv(sqrt_C)

        try:
            W_v, W_o = get_V_O_weights(model=model, layer_idx=layer)
        except Exception as e:
            logger.warning(f"[VO] Layer {layer}: cannot access v_proj/o_proj: {e}")
            continue

        for h in range(n_heads):
            # head_start_idx, head_end_idx
            head_s, head_e = h * head_dim, (h + 1) * head_dim

            try:
                # Hd = head_dims, H = hidden dimensions
                # V_head [Hd, H], O_head [H, Hd]
                # sqrt_C [H, H]

                V_head = W_v[head_s:head_e, :].to(dtype=torch.float64, device="cuda")
                O_head = W_o[:, head_s:head_e].to(dtype=torch.float64, device="cuda")

                # V_head needs to be transposed because we expect a tensor of (in_features, out_features)
                # but torch stores weights as (out_features, in_features)
                U, _S, V = torch.linalg.svd(sqrt_C @ V_head.T, full_matrices=False)

                # sqrt_C @ V_head.T [H, Hd]
                # U [H, Hd], S [Hd, Hd], V [Hd, Hd]

                S = torch.diag(_S)  # [Hd, Hd]

                A = S @ V @ O_head.T  # once again transpose O_head
                # There is marginal difference (from minimal testing) between full_matrices=False|True
                U_p, _S_p, V_p = torch.linalg.svd(A, full_matrices=True)

                S_p = torch.diag(_S_p)

                # A [Hd, H]
                # U_p [Hd, Hd], S_p [H, H], V_p [H, H]

                # [H, H] @ [H, Hd] @ [Hd, H] = [H, Hd] @ [Hd, Hd] = [H, Hd]
                compressed_v = (inv_sqrt_C @ U @ U_p)[:, :rank_i]
                # [H, H] @ [H, H]
                compressed_o = S_p[:rank_i, :rank_i] @ V_p[:rank_i, :]

                V_new = torch.zeros_like(V_head).to(dtype=W_v.dtype, device=W_v.device)
                O_new = torch.zeros_like(O_head).to(dtype=W_o.dtype, device=W_o.device)

                # Retranspose the output because we transposed the input
                V_new[:rank_i, :] = compressed_v.T
                O_new[:, :rank_i] = compressed_o.T

                W_v[head_s:head_e, :].data.copy_(V_new)
                W_o[:, head_s:head_e].data.copy_(O_new)

            except Exception as e:
                if logger:
                    logger.warning(
                        f"[VO] Layer {layer} Head {h}: compression failed: {e}"
                    )

        if logger:
            logger.info(
                f"[VO] ✅ Compressed layer {layer} to rank {rank_i} per head (λ={ridge_lambda})"
            )
        torch.cuda.empty_cache()


def get_V_O_weights(model, layer_idx):
    if hasattr(model.model, "decoder"):
        block = model.model.decoder.layers[layer_idx]
        W_v = block.self_attn.v_proj.weight
        W_o = block.self_attn.out_proj.weight
    elif hasattr(model.model, "layers"):
        block = model.model.layers[layer_idx]
        W_v = block.self_attn.v_proj.weight
        W_o = block.self_attn.o_proj.weight
    else:
        raise AttributeError

    return W_v, W_o

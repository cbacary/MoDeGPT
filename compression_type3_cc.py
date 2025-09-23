import logging

import torch
from scipy.linalg import sqrtm
from torch.types import Tensor

logger = logging.getLogger("MoDeGPT")


@torch.no_grad()
def compress_vo(
    model,
    cov: list[Tensor] = None,
    keep_ratios=None,
    rank=None,
    n_layers=None,
    n_heads=None,
    head_dim=None,
    ridge_lambda=1e-4,
    min_rank=16,
    max_condition_number=1e4,
    logger=None,
):
    for layer in range(n_layers):
        keep_ratio = keep_ratios[layer]
        rank_i = int(head_dim * keep_ratio) if rank is None else rank
        rank_i = max(min_rank, min(rank_i, head_dim))

        # this regularizes C so that we dont get the positive definite error
        # torch.eye returns a diagnol with all 1's,
        # this multiplied by ridge_lamda and added to C will make C slightly larger to not get this err
        _C = cov[layer].to(device="cpu")
        C = _C + torch.eye(_C.shape[0], device="cpu") * ridge_lambda

        # === This is the most computationally exspensive task ======
        # Can possible make cov_x a numpy array from the beginning to reduce this
        C_np = C.detach().cpu().numpy()  # should already be on cpu but just in case
        sqrt_C_np = sqrtm(C_np).real
        sqrt_C = torch.from_numpy(sqrt_C_np).to(dtype=torch.float32, device="cuda")
        inv_sqrt_C = torch.linalg.inv(sqrt_C).float()
        # ============================================================

        # sqrt_C = torch.linalg.cholesky(C).float() -- not the right way to calculate sqrt

        try:
            W_v, W_o = get_V_O_weights(model=model, layer_idx=layer)
        except Exception as e:
            logger.warning(f"[VO] Layer {layer}: cannot access v_proj/o_proj: {e}")
            continue

        for h in range(n_heads):
            # head_start_idx, head_end_idx
            head_s, head_e = h * head_dim, (h + 1) * head_dim

            try:
                V_head = W_v[head_s:head_e, :].clone().float().to("cuda")
                O_head = W_o[:, head_s:head_e].clone().float().to("cuda")
                # V_head [Hd, D] | O_head [D, Hd]

                # V_head needs to be transposed because a tensor of (in_features, out_features) but
                # torch stores weights as (out_features, in_features)
                U, _S, V = torch.linalg.svd(sqrt_C @ V_head.T, full_matrices=False)

                S = torch.diag(_S)  # [Hd, Hd]

                A = S @ V @ O_head.T  # once again transpose O_head
                U_p, _S_p, V_p = torch.linalg.svd(A, full_matrices=False)
                S_p = torch.diag(_S_p)

                compressed_v = (inv_sqrt_C @ U @ U_p)[:, :rank_i]
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

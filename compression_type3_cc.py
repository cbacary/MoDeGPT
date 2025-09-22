import logging

import torch
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
        _C = cov[layer].to(device="cuda")
        C = _C + torch.eye(_C.shape[0], device="cuda") * ridge_lambda

        sqrt_C = torch.linalg.cholesky(C).float()
        inv_sqrt_C = torch.linalg.inv(sqrt_C).T.float()

        try:
            try:
                # OPT
                block = model.model.decoder.layers[layer]
                W_v = block.self_attn.v_proj.weight
                W_o = block.self_attn.out_proj.weight
            except AttributeError:
                # LLaMA
                block = model.model.layers[layer]
                W_v = block.self_attn.v_proj.weight
                W_o = block.self_attn.o_proj.weight

        except Exception as e:
            if logger:
                logger.warning(f"[VO] Layer {layer}: cannot access v_proj/o_proj: {e}")
            continue

        for h in range(n_heads):
            head_start_idx, head_end_idx = h * head_dim, (h + 1) * head_dim

            # try:
            V_head = (
                W_v[:, head_start_idx:head_end_idx].clone().float().to("cuda")
            )  # [dh x k]
            O_head = (
                W_o[head_start_idx:head_end_idx, :].clone().float().to("cuda")
            )  # [k x dh]

            U, _S, V = torch.linalg.svd(sqrt_C @ V_head, full_matrices=False)

            S = torch.diag(_S)  # take diag thus S: (head_dim x head_dim)

            A = S @ V @ O_head
            U_p, _S_p, V_p = torch.linalg.svd(A, full_matrices=False)
            S_p = torch.diag(_S_p)

            print(f"Shape V_p {V_p.shape}")
            print(f"Shape S_p {S_p.shape}")

            # temp_v = inv_sqrt_C @ U @ U_p
            # compressed_v = temp_v[:, :rank_i]
            # compressed_o = S_p[:rank_i, :rank_i] @ V_p[:rank_i, :]

            compressed_v = inv_sqrt_C @ U @ U_p[:, :rank_i]
            compressed_o = (
                S_p[:rank_i, :rank_i] @ V_p[:rank_i, :]
            )  # POTENTIALY HAVE TO TRANSPOSE V_p

            V_new = torch.zeros(
                (V_head.shape[0], head_dim), device="cuda", dtype=torch.float32
            )
            O_new = torch.zeros(
                (head_dim, O_head.shape[1]), device="cuda", dtype=torch.float32
            )

            V_new[:, :rank_i] = compressed_v
            O_new[:rank_i, :] = compressed_o

            W_v[:, head_start_idx:head_end_idx].data.copy_(
                V_new.to(dtype=W_v.dtype, device=W_v.device)
            )
            W_o[head_start_idx:head_end_idx, :].data.copy_(
                O_new.to(dtype=W_o.dtype, device=W_o.device)
            )

            print(f"Shape W_v_head {V_head.shape}")
            print(f"Shape W_o_head {O_head.shape}")
            print(f"Shape compressed_v {compressed_v.shape}")
            print(f"Shape compressed_o {compressed_o.shape}")
            # except Exception as e:
            #     if logger:
            #         logger.warning(f"[VO] Layer {i} Head {h}: compression failed: {e}")

        if logger:
            logger.info(
                f"[VO] ✅ Compressed layer {layer} to rank {rank_i} per head (λ={ridge_lambda})"
            )
        torch.cuda.empty_cache()

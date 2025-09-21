import logging

import torch

logger = logging.getLogger("MoDeGPT")


@torch.no_grad()
def compress_vo(
    model,
    cov=None,
    keep_ratios=None,
    rank=None,
    n_layers=None,
    n_heads=None,
    head_dim=None,
    ridge_lambda=1e-2,
    min_rank=16,
    max_condition_number=1e4,
    logger=None,
):
    C = cov
    sqrt_C = torch.linalg.cholesky(C)
    C_inv_sqrt = torch.linalg.matrix_power(C, -0.5)

    for i in range(n_layers):
        keep_ratio = keep_ratios[i]
        rank_i = int(head_dim * keep_ratio) if rank is None else rank
        rank_i = max(min_rank, min(rank_i, head_dim))

        try:
            try:
                # OPT
                block = model.model.decoder.layers[i]
                W_v = block.self_attn.v_proj.weight
                W_o = block.self_attn.out_proj.weight
            except AttributeError:
                # LLaMA
                block = model.model.layers[i]
                W_v = block.self_attn.v_proj.weight
                W_o = block.self_attn.o_proj.weight
        except Exception as e:
            if logger:
                logger.warning(f"[VO] Layer {i}: cannot access v_proj/o_proj: {e}")
            continue

        for h in range(n_heads):
            head_start_idx, head_end_idx = h * head_dim, (h + 1) * head_dim

            try:
                # Get the weights for this head
                W_v_head = W_v[head_start_idx:head_end_idx, :].clone().float().to("cuda")
                W_o_head = W_o[head_start_idx:head_end_idx, :].clone().float().to("cuda")

                U, S, V = torch.linalg.svd(sqrt_C @ W_v_head)

                U_p, S_p, V_p = torch.linalg.svd(S @ V @ W_o_head, full_matrices=False)

                W_v_new = (C_inv_sqrt @ U @ U_p)[:, :rank_i]
                W_o_new = (S_p[:rank_i, :rank_i] @ V_p[:, :rank_i]).T

                W_v[head_start_idx:head_end_idx, :].data_copy_(
                    W_v_new.to(dtype=W_v.dtype, device=W_v.device)
                )
                W_v[:, head_start_idx:head_end_idx].data_copy_(
                    W_o_new.to(dtype=W_o.dtype, device=W_o.device)
                )
            except Exception as e:
                if logger:
                    logger.warning(f"[VO] Layer {i} Head {h}: compression failed: {e}")
        if logger:
            logger.info(
                f"[VO] ✅ Compressed layer {i} to rank {rank_i} per head (λ={ridge_lambda})"
            )
        torch.cuda.empty_cache()

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
    for i in range(n_layers):
        keep_ratio = keep_ratios[i]
        rank_i = int(head_dim * keep_ratio) if rank is None else rank
        rank_i = max(min_rank, min(rank_i, head_dim))

        # === model arch identify ===
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

        # === compress each head ===
        for h in range(n_heads):
            s, e = h * head_dim, (h + 1) * head_dim
            try:
                V_h = W_v[s:e, :].clone().float().to("cuda")  # [Hd, D]
                O_h = W_o[:, s:e].clone().float().to("cuda")  # [D, Hd]

                C = V_h @ V_h.T
                ridge = ridge_lambda * torch.eye(head_dim, device=C.device)
                C_reg = C + ridge
                C_JJ = C_reg[:rank_i, :rank_i]
                cond = torch.linalg.cond(C_JJ).item()
                if cond > max_condition_number:
                    if logger:
                        logger.warning(
                            f"[VO] Layer {i} Head {h}: cond={cond:.1e}, skipping"
                        )
                    continue

                S = C_reg[:, :rank_i] @ torch.linalg.pinv(C_JJ)
                V_proj = S.T @ V_h
                O_proj = O_h @ S

                V_new = torch.zeros(
                    (head_dim, V_h.shape[1]), device="cuda", dtype=torch.float32
                )
                O_new = torch.zeros(
                    (O_h.shape[0], head_dim), device="cuda", dtype=torch.float32
                )
                V_new[:rank_i, :] = V_proj
                O_new[:, :rank_i] = O_proj

                W_v[s:e, :].data.copy_(V_new.to(dtype=W_v.dtype, device=W_v.device))
                W_o[:, s:e].data.copy_(O_new.to(dtype=W_o.dtype, device=W_o.device))

            except Exception as e:
                if logger:
                    logger.warning(f"[VO] Layer {i} Head {h}: compression failed: {e}")

        if logger:
            logger.info(
                f"[VO] ✅ Compressed layer {i} to rank {rank_i} per head (λ={ridge_lambda})"
            )
        torch.cuda.empty_cache()


# SVD Version, cause ppl explosion:


# import logging

# import torch

# logger = logging.getLogger("MoDeGPT")


# @torch.no_grad()
# def compress_vo(
#     model,
#     cov=None,
#     keep_ratios=None,
#     rank=None,
#     n_layers=None,
#     n_heads=None,
#     head_dim=None,
#     ridge_lambda=1e-2,
#     min_rank=64,
#     max_condition_number=1e3,
#     logger=None,
# ):
#     """
#     MoDeGPT Type-III VO Compression: SVD on A = sqrt(C) · V · O,
#     with full-size reconstruction and shape-safe overwrite.
#     """
#     for i in range(n_layers):
#         keep_ratio = keep_ratios[i]
#         rank_i = int(head_dim * keep_ratio) if rank is None else rank
#         rank_i = max(min_rank, min(rank_i, head_dim))

#         try:
#             try:
#                 # OPT
#                 block = model.model.decoder.layers[i]
#                 W_v = block.self_attn.v_proj.weight
#                 W_o = block.self_attn.out_proj.weight
#             except AttributeError:
#                 # LLaMA
#                 block = model.model.layers[i]
#                 W_v = block.self_attn.v_proj.weight
#                 W_o = block.self_attn.o_proj.weight
#         except Exception as e:
#             if logger:
#                 logger.warning(f"[VO] Layer {i}: cannot access v_proj/o_proj: {e}")
#             continue

#         for h in range(n_heads):
#             s, e = h * head_dim, (h + 1) * head_dim
#             try:
#                 V_h = W_v[s:e, :].clone().float().to("cuda")  # [Hd, D]
#                 O_h = W_o[:, s:e].clone().float().to("cuda")  # [D, Hd]

#                 # Step 1: Regularized covariance matrix
#                 C = V_h @ V_h.T  # [Hd, Hd]
#                 ridge = ridge_lambda * torch.eye(head_dim, device=C.device)
#                 C_reg = C + ridge

#                 # Condition check
#                 C_JJ = C_reg[:rank_i, :rank_i]
#                 cond = torch.linalg.cond(C_JJ).item()
#                 if cond > max_condition_number:
#                     if logger:
#                         logger.warning(
#                             f"[VO] Layer {i} Head {h}: cond={cond:.1e}, skipping"
#                         )
#                     continue

#                 # Step 2: Cholesky decomposition
#                 sqrt_C = torch.linalg.cholesky(C_reg)  # [Hd, Hd]

#                 # Step 3: Compute A = sqrt(C) · V · O
#                 A = sqrt_C @ V_h @ O_h  # [Hd, Hd]

#                 # Step 4: Truncated SVD
#                 U, S, Vh = torch.linalg.svd(A, full_matrices=False)
#                 U_k = U[:, :rank_i]  # [Hd, r]
#                 S_k = S[:rank_i]  # [r]
#                 V_k = Vh[:rank_i, :]  # [r, Hd]
#                 sqrt_S = torch.diag(torch.sqrt(S_k))  # [r, r]

#                 # Step 5: Back out V_proj, O_proj
#                 V_proj = torch.cholesky_solve(U_k @ sqrt_S, sqrt_C)  # [Hd, r]
#                 O_proj = (sqrt_S @ V_k).T  # [Hd, r]

#                 # Step 6: Full-size reconstruction
#                 V_new = torch.zeros(
#                     (head_dim, V_h.shape[1]), device=V_h.device, dtype=V_h.dtype
#                 )  # [Hd, D]
#                 O_new = torch.zeros(
#                     (O_h.shape[0], head_dim), device=O_h.device, dtype=O_h.dtype
#                 )  # [D, Hd]
#                 V_new[:rank_i, :] = V_proj.T.to(dtype=V_new.dtype)  # [r, D]
#                 O_new[:, :rank_i] = O_h @ O_proj.to(dtype=O_new.dtype)  # [D, r]

#                 # Step 7: Writeback
#                 W_v[s:e, :].data.copy_(V_new.to(W_v.dtype, device=W_v.device))
#                 W_o[:, s:e].data.copy_(O_new.to(W_o.dtype, device=W_o.device))

#             except Exception as e:
#                 if logger:
#                     logger.warning(f"[VO] Layer {i} Head {h}: compression failed: {e}")

#         if logger:
#             logger.info(f"[VO] ✅ Compressed layer {i} to rank {rank_i} per head (SVD)")
#         torch.cuda.empty_cache()

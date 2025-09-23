# compression_type2.py


import logging
from logging import Logger

import torch
from torch.types import Tensor

from compression_utils import sqrt_M

logger = logging.getLogger("MoDeGPT")


@torch.no_grad()
def compress_qk(
    model,
    cov,
    keep_ratios,
    rank=None,
    n_layers=None,
    n_heads=None,
    head_dim=None,
    ridge_lambda=1e-2,
    logger: Logger = None,
):
    """
    MoDeGPT Type-II Compression (Q/K): Stable interpolation version using
    MoDeGPT CR scores (||C_q^{1/2}[:,i]|| * ||C_k^{1/2}[:,i]||),
    followed by row reconstruction (your original working logic).
    """
    cov_q_list, cov_k_list = cov  # List[List[Tensor]] for Q and K respectively

    for i in range(n_layers):
        try:
            keep_ratio = keep_ratios[i]
            rank_i = int(head_dim * keep_ratio) if rank is None else rank
            rank_i = max(1, min(rank_i, head_dim))

            # === Get Q, K weight reference ===
            try:
                block = model.model.decoder.layers[i]  # OPT
                W_q = block.self_attn.q_proj.weight
                W_k = block.self_attn.k_proj.weight
            except AttributeError:
                try:
                    block = model.transformer.h[i]  # GPT (unsupported)
                    raise NotImplementedError(
                        "GPT packed QKV not supported in Type-II compression."
                    )
                except AttributeError:
                    block = model.model.layers[i]  # LLaMA
                    W_q = block.self_attn.q_proj.weight
                    W_k = block.self_attn.k_proj.weight

            for h in range(n_heads):
                h_start = h * head_dim
                h_end = (h + 1) * head_dim

                C_q = cov_q_list[i][h].float().to("cuda")  # [Hd, Hd]
                C_k = cov_k_list[i][h].float().to("cuda")  # [Hd, Hd]

                # === Use MoDeGPT CR scores: ||C_q^{1/2}[:,i]|| * ||C_k^{1/2}[:,i]||
                # ridge_eye = ridge_lambda * torch.eye(head_dim, device="cuda")
                # sqrt_C_q = torch.linalg.cholesky(C_q + ridge_eye)  # [Hd, Hd]
                # sqrt_C_k = torch.linalg.cholesky(C_k + ridge_eye)
                sqrt_C_q = sqrt_M(C_q)
                sqrt_C_k = sqrt_M(C_k)

                norms_q = torch.linalg.vector_norm(sqrt_C_q, dim=0)
                norms_k = torch.linalg.vector_norm(sqrt_C_k, dim=0)
                scores = norms_q * norms_k

                topk = torch.topk(scores, k=rank_i, largest=True).indices.to(
                    dtype=torch.uint8
                )

                topk = torch.tensor(
                    [1 if j in topk else 0 for j in range(head_dim)]
                ).to(dtype=torch.bool, device="cuda")

                Q_head: Tensor = W_q[h_start:h_end, :]
                K_head: Tensor = W_k[:, h_start:h_end]

                # S_k = torch.zeros_like(Q_head)
                # S_k[:, topk] = 1

                Q_new = torch.zeros_like(Q_head).to(
                    dtype=Q_head.dtype, device=Q_head.device
                )
                K_new = torch.zeros_like(K_head).to(
                    dtype=K_head.dtype, device=K_head.device
                )

                # Q_new[:, :] = Q_head @ S_k
                # K_new[:, :] = S_k.T @ K_head

                Q_new[topk, :] = Q_head[topk, :]
                K_new[:, topk] = K_head[:, topk]

                Q_new = Q_new.to(dtype=W_q.dtype, device=W_q.device)
                K_new = K_new.to(dtype=W_k.dtype, device=W_k.device)

                W_q[h_start:h_end, :].data.copy_(Q_new)
                W_k[:, h_start:h_end].data.copy_(K_new)

            if logger:
                logger.info(
                    f"[QK] ✅ Layer {i}: compressed to rank {rank_i} per head (CR-score + interpolation)"
                )

        except Exception as e:
            if logger:
                logger.error("Error: %s", e, exc_info=True)
                logger.warning(f"[QK] Compression failed at layer {i}: {e}")

# compression_type2.py


import logging
from logging import Logger

import torch

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
                ridge_eye = ridge_lambda * torch.eye(head_dim, device="cuda")
                sqrt_C_q = torch.linalg.cholesky(C_q + ridge_eye)  # [Hd, Hd]
                sqrt_C_k = torch.linalg.cholesky(C_k + ridge_eye)

                norms_q = torch.norm(sqrt_C_q, dim=0)
                norms_k = torch.norm(sqrt_C_k, dim=0)
                scores = norms_q * norms_k

                topk = torch.topk(scores, k=rank_i, largest=True).indices
                rest = torch.tensor(
                    [j for j in range(head_dim) if j not in topk],
                    dtype=torch.long,
                    device=topk.device,
                )

                # === Q: row reconstruction
                C_q_JJ = C_q[topk][:, topk]
                ridge_q = ridge_lambda * torch.eye(rank_i, device="cuda")
                pinv_q = torch.linalg.pinv(C_q_JJ + ridge_q)
                alpha_q = C_q[rest][:, topk] @ pinv_q  # [|rest|, r]

                W_q_h = W_q[h_start:h_end, :].float().to("cuda")  # [Hd, D]
                W_q_new = torch.zeros_like(W_q_h)
                W_q_new[topk, :] = W_q_h[topk, :]
                if len(rest) > 0:
                    W_q_new[rest, :] = alpha_q @ W_q_h[topk, :]
                W_q[h_start:h_end, :].data.copy_(W_q_new.to(W_q.dtype).to(W_q.device))

                # === K: row reconstruction
                C_k_JJ = C_k[topk][:, topk]
                ridge_k = ridge_lambda * torch.eye(rank_i, device="cuda")
                pinv_k = torch.linalg.pinv(C_k_JJ + ridge_k)
                alpha_k = C_k[rest][:, topk] @ pinv_k

                W_k_h = W_k[h_start:h_end, :].float().to("cuda")  # [Hd, D]
                W_k_new = torch.zeros_like(W_k_h)
                W_k_new[topk, :] = W_k_h[topk, :]
                if len(rest) > 0:
                    W_k_new[rest, :] = alpha_k @ W_k_h[topk, :]
                W_k[h_start:h_end, :].data.copy_(W_k_new.to(W_k.dtype).to(W_k.device))

            if logger:
                logger.info(
                    f"[QK] ✅ Layer {i}: compressed to rank {rank_i} per head (CR-score + interpolation)"
                )

        except Exception as e:
            if logger:
                logger.error("Error: %s", e, exc_info=True)
                logger.warning(f"[QK] Compression failed at layer {i}: {e}")

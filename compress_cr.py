# Type 2 compression


import logging

import torch
from torch.types import Tensor

from compression_utils import slice_QK_dims, sqrt_M
from model_utils import get_model_attrs

logger = logging.getLogger("MoDeGPT")


@torch.no_grad()
def compress_qk(
    model,
    cov,
    keep_ratios,
    rank=None,
    ridge_lambda=1,
    slice_dims=True,
):
    """
    MoDeGPT Type-II Compression (Q/K): Stable interpolation version using
    MoDeGPT CR scores (||C_q^{1/2}[:,i]|| * ||C_k^{1/2}[:,i]||),
    followed by row reconstruction (your original working logic).
    """
    cov_q_list, cov_k_list = cov  # List[List[Tensor]] for Q and K respectively

    n_layers, n_heads, _, head_dim, arch = get_model_attrs(model)

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

            new_Q_heads = []
            new_K_heads = []
            for h in range(n_heads):
                h_start = h * head_dim
                h_end = (h + 1) * head_dim

                C_q = cov_q_list[i][h].to(dtype=torch.float64, device="cuda")  # [Hd, Hd]
                C_k = cov_k_list[i][h].to(dtype=torch.float64, device="cuda")  # [Hd, Hd]

                if torch.isnan(C_q).any():
                    print("Big boom problem C_q nan")
                if torch.isinf(C_q).any():
                    print("Big boom problem C_q inf")
                if torch.isnan(C_k).any():
                    print("Big boom problem C_k nan")
                if torch.isinf(C_k).any():
                    print("Big boom problem C_k inf")

                # C_q = C_q + (ridge_lambda * torch.eye(C_q.shape[0], device=C_q.device))
                # C_k = C_k + (ridge_lambda * torch.eye(C_k.shape[0], device=C_k.device))

                # === Use MoDeGPT CR scores: ||C_q^{1/2}[:,i]|| * ||C_k^{1/2}[:,i]||
                # ridge_eye = ridge_lambda * torch.eye(head_dim, device="cuda")
                # sqrt_C_q = torch.linalg.cholesky(C_q + ridge_eye)  # [Hd, Hd]
                # sqrt_C_k = torch.linalg.cholesky(C_k + ridge_eye)
                sqrt_C_q = sqrt_M(C_q)
                sqrt_C_k = sqrt_M(C_k)

                # do we have to take transpose of Q or K?
                norms_q = torch.linalg.vector_norm(sqrt_C_q, dim=0)
                norms_k = torch.linalg.vector_norm(sqrt_C_k, dim=0)
                scores = norms_q * norms_k

                # NOTE TO ME LATER::: Consider dims input into topk (does it work with vector norm above)
                topk = torch.topk(scores, k=rank_i, largest=True).indices

                Sk = torch.eye(sqrt_C_k.shape[0], device="cuda", dtype=torch.float64)[:, topk]

                topk_selector = torch.tensor([1 if j in topk else 0 for j in range(head_dim)]).to(
                    dtype=torch.bool, device="cuda"
                )

                Q_head: Tensor = W_q[h_start:h_end, :].to(device="cuda", dtype=torch.float64)
                K_head: Tensor = W_k[h_start:h_end, :].to(device="cuda", dtype=torch.float64)

                if slice_dims:
                    Q_new = Sk.T @ Q_head  # dont trust this, gotta rethink about that
                    K_new = Sk.T @ K_head
                    new_Q_heads.append(Q_new.T)
                    new_K_heads.append(K_new.T)
                else:
                    Q_new = torch.zeros_like(Q_head).to(dtype=Q_head.dtype, device=Q_head.device)
                    K_new = torch.zeros_like(K_head).to(dtype=K_head.dtype, device=K_head.device)

                    Q_new[topk_selector, :] = Q_head[topk_selector, :]
                    K_new[topk_selector, :] = K_head[topk_selector, :]

                    Q_new = Q_new.to(dtype=W_q.dtype, device=W_q.device)
                    K_new = K_new.to(dtype=W_k.dtype, device=W_k.device)

                    W_q[h_start:h_end, :].data.copy_(Q_new)
                    W_k[h_start:h_end, :].data.copy_(K_new)

                """"
                
                The original code from the MoDeGPT github i was originally given

                topk = torch.topk(scores, k=rank_i, largest=True).indices
                rest = torch.tensor([j for j in range(head_dim) if j not in topk], dtype=torch.long, device=topk.device)

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

                
                """

            if slice_dims:
                slice_QK_dims(
                    model=model,
                    layer_idx=i,
                    new_heads_Q=new_Q_heads,
                    new_heads_K=new_K_heads,
                    bias=False,
                )

            if logger:
                logger.info(
                    f"[QK] ✅ Layer {i}: compressed to rank {rank_i} per head (CR-score + interpolation)"
                )
        except Exception as e:
            if logger:
                logger.error("Error: %s", e, exc_info=True)
                logger.warning(f"[QK] Compression failed at layer {i}: {e}")

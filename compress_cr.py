# Type 2 compression


from audioop import bias
import logging

import torch
from torch.types import Tensor

from compression_utils import slice_QK_dims, sqrt_M
from model_utils import get_model_attrs

logger = logging.getLogger("MoDeGPT")


#### NOTE: see OPTDecoderLayer.final_layer_norm
@torch.no_grad()
def compress_qk_svd(
    model,
    cov_x: list[Tensor],
    keep_ratios,
    rank=None,
    ridge_lambda=1,
    slice_dims=True,
):
    """
    QK compression using SVD instead of CR decomposition
    """
    n_layers, n_heads, _, head_dim, arch = get_model_attrs(model)

    for layer in range(n_layers):
        # try:
        keep_ratio = keep_ratios[layer]
        rank_i = int(head_dim * keep_ratio) if rank is None else rank
        rank_i = max(1, min(rank_i, head_dim))

        C = cov_x[layer].to(device="cuda", dtype=torch.float64)
        sqrt_C = sqrt_M(C)
        inv_sqrt_C = torch.linalg.inv(sqrt_C)

        try:
            block = model.model.decoder.layers[layer]  # OPT
            W_q = block.self_attn.q_proj.weight
            W_k = block.self_attn.k_proj.weight
            bias_q: Tensor = block.self_attn.q_proj.bias
            bias_k: Tensor = block.self_attn.k_proj.bias
        except AttributeError:
            try:
                block = model.transformer.h[layer]  # GPT (unsupported)
                raise NotImplementedError("GPT packed QKV not supported in Type-II compression.")
            except AttributeError:
                block = model.model.layers[layer]  # LLaMA
                W_q = block.self_attn.q_proj.weight
                bias_q: Tensor = block.self_attn.q_proj.bias
                W_k = block.self_attn.k_proj.weight
                bias_k: Tensor = block.self_attn.k_proj.bias

        new_Q_heads = []
        new_K_heads = []
        bias_Q_heads = []
        bias_K_heads = []
        for h in range(n_heads):
            h_start = h * head_dim
            h_end = (h + 1) * head_dim

            Q_head: Tensor = W_q[h_start:h_end, :].to(C)
            K_head: Tensor = W_k[h_start:h_end, :].to(C)
            Q_head_bias: Tensor = bias_q[h_start:h_end].to(C)
            K_head_bias: Tensor = bias_k[h_start:h_end].to(C)

            u, s, v = torch.linalg.svd(sqrt_C @ Q_head.T, full_matrices=False)
            s = torch.diag(s)

            # print(f"u.shape = {u.shape}")
            # print(f"s.shape = {s.shape}")
            # print(f"v.shape = {v.shape}")

            u_p, s_p, v_p = torch.linalg.svd(s @ v @ K_head)

            Q = (inv_sqrt_C @ u @ u_p)[:, :rank_i]
            K = torch.diag(s_p[:rank_i]) @ v_p[:rank_i]

            # print(f"Q_head.shape = {Q_head.shape}")  # [d_model, d_model]
            # print(f"u.shape = {u.shape}")  # [d_model, d_model]
            # print(f"s.shape = {s.shape}")  # [d_model]
            # print(f"v.shape = {v.shape}")  # [d_model, d_model]
            # print(f"inv_sqrt_C.shape = {inv_sqrt_C.shape}")  # [d_model, d_model]

            # Q = inv_sqrt_C @ u[:, :rank_i]  # [d_model, rank_i]
            # K = v[:, :rank_i] @ torch.diag_embed(s[:rank_i])  # [d_model, rank_i]
            alpha = torch.sqrt(torch.abs(K).max() / torch.abs(Q).max())  # make Q K of similar scale
            # print(f"Q.shape = {Q.shape}")  # [d_model, d_model]
            Q = Q * alpha
            K /= alpha

            # print(f"Q.shape = {Q.shape}")
            # print(f"K.shape = {K.shape}")

            Q = Q.T  # [rank_i, d_model]
            K = K  # [rank_i, d_model]

            new_Q_heads.append(Q)
            new_K_heads.append(K)

            # Q.T [d_model, rank_i], Q_head.T [head_dim, ]
            # bias_proj_Q = Q_head.T @ Q_head_bias.view(-1, 1)
            # bias_proj_K = K_head.T @ K_head_bias.view(-1, 1)
            # print(f"bias_proj_Q.shape = {bias_proj_Q.shape}")
            # print(f"bias_proj_K.shape = {bias_proj_K.shape}")
            # new_bias_Q = torch.linalg.solve(Q, bias_proj_Q).view(-1)
            # new_bias_K = torch.linalg.solve(K, bias_proj_K).view(-1)

            scale = alpha * torch.diag_embed(torch.reciprocal(s_p[:rank_i]))  #
            # scale = 1 / alpha * torch.diag_embed(torch.reciprocal(s_p[:rank_i])) # makes scale too small, bias goes to 0
            print(f"alpha = {alpha}")
            print(f"scale.mean() = {scale.mean()}")
            # print(f"scale.shape = {scale.shape}")
            # print(f"u_p.shape = {u_p.shape}")
            # print(f"v_p.shape = {v_p.shape}")
            # [rank_i, rank_i] @ [rank_i, d_model] @ [d_model, rank_i]
            new_bias_Q: Tensor = scale @ v_p[:, :rank_i].T @ K_head.T @ Q_head_bias
            print(f"old_bias_Q.mean = {Q_head_bias.mean(dim=0)}")
            print(f"new_bias_Q.mean = {new_bias_Q.mean(dim=0)}")

            # new_bias_Q = (torch.pinverse(Q.T) @ Q_head.T @ Q_head_bias.view(-1, 1)).view(-1) # also makes bias go to zero
            new_bias_K = (torch.pinverse(K.T) @ K_head.T @ K_head_bias.view(-1, 1)).view(
                -1
            )  # also makes bias go to zero

            bias_Q_heads.append(new_bias_Q)
            bias_K_heads.append(new_bias_K)

            # Q_new = Q_head.T @ Sk  # dont trust this, gotta rethink about that
            # K_new = K_head.T @ Sk
            # bias_q_new = bias_q[h_start:h_end][topk_selector]
            # bias_k_new = bias_k[h_start:h_end][topk_selector]
            # new_Q_heads.append(Q_new.T)
            # new_K_heads.append(K_new.T)
            # bias_Q_heads.append(bias_q_new)
            # bias_K_heads.append(bias_k_new)

        ####

        slice_QK_dims(
            model=model,
            layer_idx=layer,
            new_heads_Q=new_Q_heads,
            new_heads_K=new_K_heads,
            new_bias_Q=bias_Q_heads,
            new_bias_K=bias_K_heads,
            bias=True,
        )

        if logger:
            logger.info(
                f"[QK] ✅ Layer {layer}: compressed to rank {rank_i} per head (CR-score + interpolation)"
            )


#### NOTE: see OPTDecoderLayer.final_layer_norm
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
        # try:
        keep_ratio = keep_ratios[i]
        rank_i = int(head_dim * keep_ratio) if rank is None else rank
        rank_i = max(1, min(rank_i, head_dim))

        try:
            block = model.model.decoder.layers[i]  # OPT
            W_q = block.self_attn.q_proj.weight
            W_k = block.self_attn.k_proj.weight
            bias_q = block.self_attn.q_proj.bias
            bias_k = block.self_attn.k_proj.bias
        except AttributeError:
            try:
                block = model.transformer.h[i]  # GPT (unsupported)
                raise NotImplementedError("GPT packed QKV not supported in Type-II compression.")
            except AttributeError:
                block = model.model.layers[i]  # LLaMA
                W_q = block.self_attn.q_proj.weight
                bias_q = block.self_attn.q_proj.bias
                W_k = block.self_attn.k_proj.weight
                bias_k = block.self_attn.k_proj.bias

        new_Q_heads = []
        new_K_heads = []
        bias_Q_heads = []
        bias_K_heads = []
        for h in range(n_heads):
            h_start = h * head_dim
            h_end = (h + 1) * head_dim

            C_q = cov_q_list[i][h].to(dtype=torch.float64, device="cuda")  # [Hd, Hd]
            C_k = cov_k_list[i][h].to(dtype=torch.float64, device="cuda")  # [Hd, Hd]

            sqrt_C_q = sqrt_M(C_q)
            sqrt_C_k = sqrt_M(C_k)

            # symmetric matrix, dim doesnt matter
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

            Q_new = Q_head.T @ Sk  # dont trust this, gotta rethink about that
            K_new = K_head.T @ Sk
            bias_q_new = bias_q[h_start:h_end][topk_selector]
            bias_k_new = bias_k[h_start:h_end][topk_selector]
            new_Q_heads.append(Q_new.T)
            new_K_heads.append(K_new.T)
            bias_Q_heads.append(bias_q_new)
            bias_K_heads.append(bias_k_new)

        ####

        slice_QK_dims(
            model=model,
            layer_idx=i,
            new_heads_Q=new_Q_heads,
            new_heads_K=new_K_heads,
            new_bias_Q=bias_Q_heads,
            new_bias_K=bias_K_heads,
            bias=True,
        )

        if logger:
            logger.info(
                f"[QK] ✅ Layer {i}: compressed to rank {rank_i} per head (CR-score + interpolation)"
            )
    # except Exception as e:
    #     if logger:
    #         logger.error("Error: %s", e, exc_info=True)
    #         logger.warning(f"[QK] Compression failed at layer {i}: {e}")

# Type 2 compression


import logging

import torch
from torch.types import Tensor

from compression_utils import slice_QK_dims, sqrt_M
from model_utils import get_model_attrs, dtype_p, d1, d2
from patchers.LlamaRebuild import LlamaForCausalLM

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
    QK compression using SVD instead of CR decomposition -- better performance for OPT models
    """
    n_layers, n_heads, _, head_dim, arch = get_model_attrs(model)

    for layer in range(n_layers):
        # try:
        keep_ratio = keep_ratios[layer]
        rank_i = int(head_dim * keep_ratio) if rank is None else rank
        rank_i = max(1, min(rank_i, head_dim))

        C = cov_x[layer].to(device="cuda", dtype=dtype_p)
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
    model: LlamaForCausalLM,
    cov,
    keep_ratios,
    rank=None,
    ridge_lambda=1,
    slice_dims=True,
):
    cov_q_list, cov_k_list = cov  # List[List[Tensor]] for Q and K respectively

    n_layers, n_heads, _, head_dim, arch = get_model_attrs(model)

    rotary_masks = []
    for i in range(n_layers):
        # try:
        keep_ratio = keep_ratios[i]
        rank_i = int(head_dim * keep_ratio) if rank is None else rank
        rank_i = max(1, min(rank_i, head_dim))

        if arch == "opt":
            block = model.model.decoder.layers[i]  # OPT
            W_q = block.self_attn.q_proj.weight
            W_k = block.self_attn.k_proj.weight
            bias_q = block.self_attn.q_proj.bias
            bias_k = block.self_attn.k_proj.bias
            bias = True
        elif arch == "llama":
            block = model.model.layers[i]  # LLaMA
            W_q = block.self_attn.q_proj.weight
            W_k = block.self_attn.k_proj.weight
            bias = False

        new_Q_heads = []
        new_K_heads = []
        bias_Q_heads = []
        bias_K_heads = []
        layer_rotary_mask = []
        for h in range(n_heads):
            h_start = h * head_dim
            h_end = (h + 1) * head_dim

            Q_head: Tensor = W_q[h_start:h_end, :].to(device=d2, dtype=torch.float64)
            K_head: Tensor = W_k[h_start:h_end, :].to(device=d2, dtype=torch.float64)
            if arch == "opt":
                Q_head_bias: Tensor = bias_q[h_start:h_end]
                K_head_bias: Tensor = bias_k[h_start:h_end]

            C_q = cov_q_list[i][h].to(dtype=torch.float64, device=d2)  # [Hd, Hd]
            C_k = cov_k_list[i][h].to(dtype=torch.float64, device=d2)  # [Hd, Hd]

            C_q += ridge_lambda * torch.eye(C_q.shape[0], device=C_q.device, dtype=C_q.dtype)
            C_k += ridge_lambda * torch.eye(C_k.shape[0], device=C_k.device, dtype=C_k.dtype)

            sqrt_C_q = sqrt_M(C_q)
            sqrt_C_k = sqrt_M(C_k)

            if arch == "llama":
                new_Q_head, new_K_head, rotary_mask_head = compress_head_llama(
                    sqrt_C_q, sqrt_C_k, Q_head, K_head, rank_i
                )
                layer_rotary_mask.append(rotary_mask_head)
            elif arch == "opt":
                new_Q_head, new_K_head, qb_h, kb_h = compress_head_opt(
                    sqrt_C_q, sqrt_C_k, Q_head, K_head, Q_head_bias, K_head_bias, rank_i
                )
                bias_Q_heads.append(qb_h)
                bias_K_heads.append(kb_h)
            else:
                raise Exception

            new_Q_heads.append(new_Q_head)
            new_K_heads.append(new_K_head)
        ####

        if arch == "llama":

            # layer_rotary_mask     [n_heads * new_head_dims] (unfolded)
            # .reshape(n_heads, -1) [n_heads, new_head_dims]
            # .unsqueeze(0)         [1, n_heads, new_head_dims]
            # .unsqueeze(2)         [1, n_heads, 1, new_head_dims]
            # this is then shapes it into the applicable shape for apply_rotary_pos_emb
            # (see comment about unsqueezing there): [batch_size, heads, seq_len, head_dim]
            final = torch.tensor([], dtype=torch.int64, device="cuda")
            for mask_head in layer_rotary_mask:
                final = torch.cat((final, mask_head.to(device="cuda", dtype=torch.int64)), dim=0)
            
            final = final.reshape(n_heads, -1).unsqueeze(0).unsqueeze(2)
            rotary_masks.append(final)
            # model.model.layers[i].self_attn.layer_rotary_mask = final.to(device=W_q.device)

            # final = final.reshape(n_heads, -1).unsqueeze(0).unsqueeze(2)

        slice_QK_dims(
            model=model,
            layer_idx=i,
            new_heads_Q=new_Q_heads,
            new_heads_K=new_K_heads,
            new_bias_Q=bias_Q_heads,
            new_bias_K=bias_K_heads,
            bias=bias,
        )

        if logger:
            logger.info(
                f"[QK] ✅ Layer {i}: compressed to rank {rank_i} per head (CR-score + interpolation)"
            )

    return rotary_masks
    # except Exception as e:
    #     if logger:
    #         logger.error("Error: %s", e, exc_info=True)
    #         logger.warning(f"[QK] Compression failed at layer {i}: {e}")


def compress_head_llama(
    sqrt_C_q: Tensor,
    sqrt_C_k: Tensor,
    Q_head: Tensor,
    K_head: Tensor,
    rank: int,
):

    # llama requires head dim divisible by two
    rank = rank - (rank % 2)
    head_dims = Q_head.shape[0]

    normed_q_r1 = torch.norm(sqrt_C_q[..., : head_dims // 2], dim=0)  # norm for query rotary half 1
    normed_q_r2 = torch.norm(sqrt_C_q[..., head_dims // 2 :], dim=0)
    normed_k_r1 = torch.norm(sqrt_C_k[..., : head_dims // 2], dim=0)
    normed_k_r2 = torch.norm(sqrt_C_k[..., head_dims // 2 :], dim=0)  # norm for key rotary half 2

    final_norm = normed_q_r1**2 * normed_k_r1**2 + normed_q_r2**2 * normed_k_r2**2
    final_norm = torch.sqrt(final_norm)
    final_norm /= final_norm.sum()

    topk = torch.topk(final_norm, k=rank // 2).indices
    Sk_mask = torch.cat((topk, topk + rank // 2))

    Sk = torch.eye(head_dims, device=d2, dtype=dtype_p)[:, Sk_mask]

    new_Q_head = Q_head.T @ Sk
    new_K_head = K_head.T @ Sk
    new_Q_head = new_Q_head.T
    new_K_head = new_K_head.T
    
    # new_Q_head = Q_head[Sk_mask].to(Q_head).to(device="cpu", dtype=torch.float16)
    # new_K_head = K_head[Sk_mask].to(K_head).to(device="cpu", dtype=torch.float16)

    return new_Q_head, new_K_head, Sk_mask


def compress_head_opt(
    sqrt_C_q: Tensor,
    sqrt_C_k: Tensor,
    Q_head: Tensor,
    K_head: Tensor,
    bias_Q_head: Tensor,
    bias_K_head: Tensor,
    rank: int,
):
    # symmetric matrix, dim doesnt matter
    norms_q = torch.linalg.vector_norm(sqrt_C_q, dim=0)
    norms_k = torch.linalg.vector_norm(sqrt_C_k, dim=0)
    scores = norms_q * norms_k

    # NOTE TO ME LATER::: Consider dims input into topk (does it work with vector norm above)
    topk = torch.topk(scores, k=rank).indices

    Q_new = Q_head[topk].to(device="cpu")
    K_new = K_head[topk].to(device="cpu")

    bias_q_new = bias_Q_head[topk]
    bias_k_new = bias_K_head[topk]

    return Q_new, K_new, bias_q_new, bias_k_new

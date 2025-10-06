# Type 1 compression
import logging

import torch

from compression_utils import slice_gate_dims
from model_utils import get_model_attrs

logger = logging.getLogger("MoDeGPT")


@torch.no_grad()
def compress_mlp(model, cov, keep_ratios, ridge_lambda=1e-4, slice_dims=True):
    """
    MoDeGPT Type-I Compression (MLP): Full Nyström approximation with ridge leverage scoring.
    """
    n_layers, _, _, _, _ = get_model_attrs(model)

    for i in range(n_layers):
        # Step 3: get weights
        try:
            block = model.model.decoder.layers[i]  # OPT
            W_u = block.fc1.weight  # [D_int, D_h]
            W_d = block.fc2.weight  # [D_h, D_int]
            bias_u = block.fc1.bias
            bias_d = block.fc1.bias
            proj_u = block.fc1
            proj_d = block.fc2
        except AttributeError:
            block = model.transformer.h[i]  # GPT
            proj_u = block.fc1
            proj_d = block.fc2
        except AttributeError:
            try:
                block = model.transformer.h[i]  # GPT
                W_u = block.mlp.c_fc.weight
                W_d = block.mlp.c_proj.weight
                proj_u = block.mlp.c_fc
                proj_d = block.mlp.c_proj
            except AttributeError:
                block = model.model.layers[i]  # LLaMA
                proj_u = block.mlp.gate_proj
                proj_d = block.mlp.down_proj
                W_u = proj_u.weight
                W_d = proj_d.weight

        keep_ratio = keep_ratios[i]
        C = cov[i].to(dtype=torch.float64, device="cuda")  # [D_int, D_int]
        D_int = C.shape[0]
        rank_i = int(D_int * keep_ratio)
        rank_i = max(1, min(rank_i, D_int))

        C_ridge = C + (ridge_lambda * torch.eye(D_int, device=C.device))
        s_i = torch.linalg.solve(C_ridge.T, C.T).T
        # s_i = inv_term @ C
        scores = torch.diag(s_i)

        topk = torch.topk(scores, k=rank_i, largest=True, dim=0).indices
        topk_selector = torch.tensor([1 if j in topk else 0 for j in range(D_int)]).to(
            dtype=torch.bool, device="cuda"
        )

        Sk = torch.eye(D_int, device=C.device, dtype=torch.float64)[
            :, topk_selector
        ]  # [D_int, rank_i]

        W_u = W_u.to(dtype=torch.float64, device="cuda")  # [D_int, D_h]
        W_d = W_d.to(dtype=torch.float64, device="cuda")  # [D_h, D_int]

        # W_u.T @ Sk =
        # = [D_h, D_int] @ [D_int, rank_i]
        # = [D_h, rank_i]
        W_u_proj = W_u.T @ Sk
        new_bias_u = bias_u[topk_selector]

        # Sk.T @ (C @ Sk) =
        # = [rank_i, D_int] @ ( [D_int, D_int] @ [D_int, rank_i] )
        # = [rank_i, D_int] @ [D_int, rank_i]
        # = [rank_i, rank_i]
        C_Sk_proj = torch.linalg.pinv(Sk.T @ (C @ Sk))  # [rank_i, rank_i]

        # Sk.T @ (C @ W_d.T) =
        # = [rank_i, D_int] @ ( [D_int, D_int] @ [D_int, D_h] )
        # = [rank_i, D_int] @ ( [D_int, D_h] )
        # = [rank_i, D_h]
        down_SK_proj = Sk.T @ (C @ W_d.T)  # as in paper
        # down_SK_proj = Sk.T @ C  # modified

        # C_Sk_proj @ down_SK_proj =
        # = [rank_i, rank_i] @ [rank_i, D_h]
        # = [rank_i, D_h]
        W_d_proj = C_Sk_proj @ down_SK_proj  # as in paper

        if slice_dims:
            slice_gate_dims(
                model=model,
                layer_idx=i,
                up_weights=W_u_proj.T,
                down_weights=W_d_proj.T,
                new_bias_u=new_bias_u,
                bias=True,
            )
        else:
            ### ignore this part
            proj_u.weight.data.zero_()
            proj_u.weight.data[:rank_i, :] = W_u_proj.to(proj_u.weight)

            proj_d.weight.data.zero_()
            proj_d.weight.data[:, :rank_i] = W_d_proj.to(proj_d.weight)
            # W_u.data.zero_()
            # W_u.data[:r, :] = W_u_proj.to(W_u.dtype).to(W_u.device)

            # W_d.data.zero_()
            # W_d.data[:, :r] = W_d_proj.to(W_d.dtype).to(W_d.device)

        if logger:
            logger.info(
                f"[MLP] ✅ Layer {i}: compressed to rank {rank_i} (Nyström λ={ridge_lambda})"
            )

        torch.cuda.empty_cache()

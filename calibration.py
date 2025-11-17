import numpy as np
import random
import logging

import torch
from torch.types import Tensor
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from typing import Tuple

from model_utils import get_model_attrs, dtype_p, calib_device

logger = logging.getLogger("MoDeGPT")

np.random.seed(1234)
random.seed(1234)


print("newest")


def load_calibs(
    model,
    tokenizer,
    texts,
    batch_size: int,
    load_calibs_from="",
    calibs_save_path="",
):
    def recursive_cast(obj, dtype, device):
        """
        Cause i was stupid, the tensors (wrapped in many, many python arrays) 
        have to be recast like this. ya live ya learn
        """
        if isinstance(obj, torch.Tensor):
            return obj.to(dtype=dtype, device=device)
        elif isinstance(obj, dict):
            return {k: recursive_cast(v, dtype, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_cast(elem, dtype, device) for elem in obj]
        elif isinstance(obj, tuple):
            return tuple(recursive_cast(elem, dtype, device) for elem in obj)
        else:
            return obj
    if not load_calibs_from:
        logger.info("Calibrating model...")
        cov_mlp, cov_q, cov_k, cov_x, bi_scores = __calibrate_model(
            model, tokenizer, texts, batch_size=batch_size
        )
    else:
        covs: dict = torch.load(load_calibs_from, map_location=torch.device("cpu"))
        # covs = torch.load(load_calibs_from, device=calib_device)

        # covs.pop("cov_mlp")
        covs = recursive_cast(covs, dtype_p, calib_device)

        # cov_mlp = covs["cov_mlp"]
        cov_mlp = None
        cov_q = covs["cov_q"]
        cov_k = covs["cov_k"]
        cov_x = covs["cov_x"]
        bi_scores = covs["bi_scores"]
        
        

    if calibs_save_path:
        covs = {
            # "cov_mlp": cov_mlp,
            "cov_q": cov_q,
            "cov_k": cov_k,
            "cov_x": cov_x,
            "bi_scores": bi_scores,
        }

        torch.save(covs, calibs_save_path)

    return cov_mlp, cov_q, cov_k, cov_x, bi_scores


@torch.no_grad()
def __calibrate_model(
    model: torch.nn.Module,
    tokenizer,
    texts,
    batch_size: int,
):
    print("hello from __calibrate_model")
    logger.info("Calibrating model")
    n_layers, n_heads, d_model, head_dim, arch = get_model_attrs(model)

    if not getattr(model.config, "output_hidden_states", False):
        model.config.output_hidden_states = True

    if arch == "gpt":
        transformer_blocks = model.transformer.h
    elif arch == "opt":
        transformer_blocks = model.model.decoder.layers
    elif arch == "llama":
        transformer_blocks = model.model.layers
    else:
        raise RuntimeError("Unsupported model architecture")

    if logger:
        logger.info(f"Detected architecture: {arch}")

    def get_inner(transformer_block):
        if arch == "gpt":
            n_inner = transformer_blocks.mlp.c_fc.out_features
        if arch == "opt":
            n_inner = transformer_block.fc1.out_features
        elif arch == "llama":
            n_inner = transformer_block.mlp.gate_proj.out_features
            logger.info(f"n_inner = {n_inner}")

        return n_inner

    # store these on the cpu otherwise big boom on gpu
    # cov_mlp_list = [
    #     torch.zeros(
    #         get_inner(transformer_blocks[i]),
    #         get_inner(transformer_blocks[i]),
    #         dtype=dtype_p,
    #         device=calib_device,
    #     )
    #     for i in range(n_layers)
    # ]
    cov_mlp_list = None
    cov_q_list = [
        [torch.zeros(head_dim, head_dim, dtype=dtype_p, device=calib_device) for _ in range(n_heads)]
        for _ in range(n_layers)
    ]
    cov_k_list = [
        [torch.zeros(head_dim, head_dim, dtype=dtype_p, device=calib_device) for _ in range(n_heads)]
        for _ in range(n_layers)
    ]
    # correlation input for type 3 compression, with d_h x d_h shape (hidden dimension x hidden dimension)
    cov_x_list = [torch.zeros(d_model, d_model, dtype=dtype_p, device=calib_device) for _ in range(n_layers)]

    bi_scores = [0.0 for _ in range(n_layers)]

    logger.info(f"len(transformer_blocks) = {len(transformer_blocks)}")
    handles = []
    # mlp_weights: list[torch.nn.Linear] = []
    for i, block in enumerate(transformer_blocks):
        if arch == "opt":
            # mlp_weights.append((block.fc1, block.activation_fn))
            # handles.append(block.fc1.register_forward_hook(_make_fc_hook(i, cov_mlp_list, logger)))
            handles.append(
                block.self_attn.q_proj.register_forward_hook(
                    _make_proj_hook(i, cov_q_list, n_heads, head_dim, d_model, logger)
                )
            )
            handles.append(
                block.self_attn.k_proj.register_forward_hook(
                    _make_proj_hook(i, cov_k_list, n_heads, head_dim, d_model, logger)
                )
            )
        elif arch == "llama":
            # For llama MLP, i think we can pre-register the hook on down projection,
            # see line 155 modeling llama. the input to pre_forward hook would map to:
            # σ_s(X @ W_U ) := X @ Wu * σ_g (X @ Wg)
            # handles.append(
            #     block.mlp.down_proj.register_forward_pre_hook(
            #         llama_pre_gate_hook(i, cov_mlp_list, logger)
            #     )
            # )
            # handles.append(
            #     block.self_attn.register_forward_hook(
            #         make_llama_attn_hook(
            #             layer_idx=i,
            #             cov_q_list=cov_q_list,
            #             cov_k_list=cov_k_list,
            #             logger=logger,
            #         ),
            #         with_kwargs=True,
            #     )
            # )
            handles.append(
                block.input_layernorm.register_forward_hook(
                    input_hook(
                        layer_idx=i,
                        cov_list=cov_x_list,
                        n_heads=n_heads,
                        head_dim=head_dim,
                        d_model=d_model,
                        logger=logger,
                    )
                )
            )
            handles.append(
                block.self_attn.k_proj.register_forward_hook(
                    _make_proj_hook(
                        layer_idx=i,
                        cov_list=cov_k_list,
                        n_heads=n_heads,
                        head_dim=head_dim,
                        d_model=d_model,
                        logger=logger,
                    )
                )
            )
            handles.append(
                block.self_attn.q_proj.register_forward_hook(
                    _make_proj_hook(
                        layer_idx=i,
                        cov_list=cov_q_list,
                        n_heads=n_heads,
                        head_dim=head_dim,
                        d_model=d_model,
                        logger=logger,
                    )
                )
            )

    model.eval()
    n_texts = 0
    for count, batch in enumerate(texts):
        n_texts += len(texts)
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(device="cuda")
        print(f"len(texts) = {len(texts)}")
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        for l in range(n_layers):
            x_in: Tensor = hidden_states[l].to(dtype_p)  # [B, T, D]
            x_out = hidden_states[l + 1].to(dtype_p)  # [B, T, D]

            bi_scores[l] += (
                torch.sum((1 - torch.cosine_similarity(x_in, x_out, dim=2)), dim=0).mean().item()
            )
            if arch == "opt":
                cov_x_list[l] += torch.sum(x_in.mT @ x_in, dim=0).to(device=calib_device)

        logger.info(f"Completed {count + 1} of {len(texts)} batches")
    #####

    for h in handles:
        h.remove()

    for layer in range(n_layers):
        bi_scores[layer] /= n_texts
        cov_x_list[layer] /= n_texts
        # cov_mlp_list[i] /= n_tokens
        # for h in range(n_heads):
        #     cov_q_list[layer][h] /= n_texts
        #     cov_k_list[layer][h] /= n_texts

    if logger:
        logger.info("Finished calibration and computed BI scores.")
    return cov_mlp_list, cov_q_list, cov_k_list, cov_x_list, bi_scores


@torch.no_grad()
def llama_pre_gate_hook(layer_idx, cov_mlp_list, logger=None):
    def hook(module, input: Tuple[torch.Tensor]):
        x_input = input[0]
        H = x_input.to(device=calib_device).detach().to(dtype=dtype_p).view(-1, x_input.size(-1))
        cov_mlp_list[layer_idx] += (H.T @ H).to(device=calib_device)

        return None

    return hook


@torch.no_grad()
def _make_fc_hook(layer_idx, cov_mlp_list, logger=None):
    def hook(module: torch.nn.Linear, inp, out):
        # out [B*T, D_int]
        # reallly we should grab the activation function directly
        # from the model its something like model.decoder....act_fn()
        act = torch.nn.functional.relu(out.to(dtype=dtype_p, device=calib_device))
        H = act.detach().to(dtype=dtype_p).view(-1, act.size(-1))
        cov_mlp_list[layer_idx] += (H.T @ H).to(device=calib_device)

    return hook

@torch.no_grad()
def make_llama_attn_hook(layer_idx, cov_q_list, cov_k_list, logger=None):
    from patchers.LlamaRebuild import LlamaAttention

    def hook(module: LlamaAttention, args, kwargs: dict, out):
        
        hidden_states: Tensor = kwargs["hidden_states"]
        position_embeddings: tuple[Tensor, Tensor] = kwargs["position_embeddings"]

        input_shape = hidden_states.shape[:-1]  # (batch_size, seq_len)
        bsz, seq_len, d_model = hidden_states.shape

        n_heads, head_dim = module.config.num_attention_heads, module.head_dims

        # hopefully this isn't neccecary
        def calc_roped_qk():
            q_proj, k_proj = module.q_proj, module.k_proj

            query_states = q_proj(hidden_states)  # [batch_size, seq_len, d_model]
            key_states = k_proj(hidden_states)  # [batch_size, seq_len, d_model]

            query_states = query_states.view((*input_shape, -1, head_dim)).transpose(1, 2)
            key_states = key_states.view((*input_shape, -1, head_dim)).transpose(1, 2)
            # qk: [batch_size, n_heads, seq_len, head_dims] .. (after transpose 1,2)

            cos, sin = position_embeddings

            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            
            # qk: [batch_size, n_heads, seq_len, head_dims]
            return query_states, key_states

        query_states = module.roped_query_states
        key_states = module.roped_key_states

        query_states: Tensor = query_states.transpose(1, 2)
        key_states: Tensor = key_states.transpose(1, 2)
        # qk: [batch_size, seq_len, n_heads, head_dims]

        query_states = query_states.view(bsz * seq_len, n_heads, head_dim)
        key_states = key_states.view(bsz * seq_len, n_heads, head_dim)
        # qk: [B*T, n_heads, head_dims]

        query_states = query_states.permute(1, 0, 2).to(dtype=dtype_p)
        key_states = key_states.permute(1, 0, 2).to(dtype=dtype_p)
        # qk: [n_heads, B*T, head_dims]

        # [n_heads, head_dims, B*T] @ [n_heads, B*T, head_dims]
        # = [n_heads, head_dims, head_dims]
        query_covariance = torch.bmm(query_states.transpose(1, 2), query_states).to(device=calib_device)
        key_covariance = torch.bmm(key_states.transpose(1, 2), key_states).to(device=calib_device)
        # qk_cov[n_heads, head_dims, head_dims]

        count = 0
        for cov_q_h, cov_k_h in zip(query_covariance, key_covariance):
            cov_q_list[layer_idx][count] += cov_q_h
            cov_k_list[layer_idx][count] += cov_k_h
            
            if not torch.isfinite(cov_q_h).any():
                print(f"cov_q_h at {layer_idx} head {count} has nan/inf")
                raise Exception("not good q")
            if not torch.isfinite(cov_k_h).any():
                print(f"cov_k_h at {layer_idx} head {count} has nan/inf")
                raise Exception("not good k")

        del module.roped_query_states
        del module.roped_key_states

    return hook


@torch.no_grad()
def input_hook(layer_idx, cov_list, n_heads, head_dim, d_model, logger=None):
    def hook(module: torch.nn.Linear, inp, out):
        proj_out = out.detach().to(dtype=dtype_p, device="cuda")  # [B,T, d_model]
        cov_list[layer_idx] += torch.sum(proj_out.mT @ proj_out, dim=0).to(device=calib_device)

    return hook

@torch.no_grad()
def _make_proj_hook(layer_idx, cov_list, n_heads, head_dim, d_model, logger=None):
    def hook(module: torch.nn.Linear, inp, out):
        proj_out = out.detach().to(dtype=dtype_p, device="cuda")  # [B,T, d_model]
        proj = proj_out.view(-1, d_model)  # [B*T, d_model]
        proj = proj_out.view(-1, n_heads, head_dim) # [B*T, n_heads, head_dim]
        proj = proj.permute(1, 0, 2) # [n_heads, B*T, head_dim]
        C_proj = torch.bmm(proj.transpose(1,2), proj).to(calib_device) # [n_heads, head_dim, head_dim]
        # C_proj = proj.T @ proj
        # C_proj = C_proj
        for h in range(n_heads):
            cov_list[layer_idx][h] += C_proj[h, :, :]
            # h_proj = C_proj[h * head_dim : (h + 1) * head_dim, h * head_dim : (h + 1) * head_dim]
            # cov_list[layer_idx][h] += h_proj.to(device=calib_device)

    return hook

import numpy as np
import random
import logging

import torch
from torch.nn.functional import normalize
from torch.types import Tensor


from model_utils import get_model_attrs

logger = logging.getLogger("MoDeGPT")

np.random.seed(1234)
random.seed(1234)


def load_calibs(
    model,
    tokenizer,
    texts,
    batch_size: int,
    load_calibs_from="",
    calibs_save_path="",
):
    if not load_calibs_from:
        logger.info("Calibrating model...")
        cov_mlp, cov_q, cov_k, cov_x, bi_scores = __calibrate_model(
            model, tokenizer, texts, batch_size=batch_size
        )
    else:
        covs = torch.load(load_calibs_from)

        cov_mlp = covs["cov_mlp"]
        cov_q = covs["cov_q"]
        cov_k = covs["cov_k"]
        cov_x = covs["cov_x"]
        bi_scores = covs["bi_scores"]

    if calibs_save_path:
        covs = {
            "cov_mlp": cov_mlp,
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

        return n_inner

    # store these on the cpu otherwise big boom on gpu
    cov_mlp_list = [
        torch.zeros(
            get_inner(transformer_blocks[i]),
            get_inner(transformer_blocks[i]),
            dtype=torch.float64,
        )
        for i in range(n_layers)
    ]
    cov_q_list = [
        [torch.zeros(head_dim, head_dim, dtype=torch.float64) for _ in range(n_heads)]
        for _ in range(n_layers)
    ]
    cov_k_list = [
        [torch.zeros(head_dim, head_dim, dtype=torch.float64) for _ in range(n_heads)]
        for _ in range(n_layers)
    ]
    # correlation input for type 3 compression, with d_h x d_h shape (hidden dimension x hidden dimension)
    cov_x_list = [torch.zeros(d_model, d_model, dtype=torch.float64) for _ in range(n_layers)]

    bi_scores = [0.0 for _ in range(n_layers)]

    logger.info(f"len(transformer_blocks) = {len(transformer_blocks)}")
    handles = []
    # mlp_weights: list[torch.nn.Linear] = []
    for i, block in enumerate(transformer_blocks):
        if arch == "gpt":
            handles.append(
                block.mlp.c_fc.register_forward_hook(_make_fc_hook(i, cov_mlp_list, logger))
            )
            handles.append(
                block.attn.c_attn.register_forward_hook(
                    _make_attn_hook(i, cov_q_list, cov_k_list, d_model, n_heads, head_dim, logger)
                )
            )
        elif arch == "opt":
            # mlp_weights.append((block.fc1, block.activation_fn))
            handles.append(block.fc1.register_forward_hook(_make_fc_hook(i, cov_mlp_list, logger)))
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
            handles.append(
                block.mlp.gate_proj.register_forward_hook(_make_fc_hook(i, cov_mlp_list, logger))
            )
            handles.append(
                block.self_attn.q_proj.register_forward_hook(
                    _make_proj_hook(i, cov_q_list, n_heads, head_dim, logger)
                )
            )
            handles.append(
                block.self_attn.k_proj.register_forward_hook(
                    _make_proj_hook(i, cov_k_list, n_heads, head_dim, logger)
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
            x_in: Tensor = hidden_states[l].to(torch.float64)  # [B, T, D]
            x_out = hidden_states[l + 1].to(torch.float64)  # [B, T, D]

            # testing showed equivalent to cosine_similary(x_in, x_out)
            # bi_scores[l] = get_BI_score(x_in, x_out)
            bi_scores[l] += (
                torch.sum((1 - torch.cosine_similarity(x_in, x_out, dim=2)), dim=0).mean().item()
            )
            cov_x_list[l] += torch.sum(x_in.mT @ x_in, dim=0).to(device="cpu")

            # for batch_i in range(len(batch)):
            # can be vectorized but thats not probably not a
            # batch_x_in = x_in[batch_i, :, :]
            # batch_x_out = x_out[batch_i, :, :]
            # bi_scores[l] = (
            #     1 - torch.cosine_similarity(batch_x_in, batch_x_out, dim=1).mean().item()
            # )
            # cov_x_list[l] += (batch_x_in.T @ batch_x_in).to(device="cpu")

        logger.info(f"Completed {count + 1} of {len(texts)} batches")
    #####

    for h in handles:
        h.remove()

    # cov_mlp_list[i] /= n_tokens
    for layer in range(n_layers):
        bi_scores[layer] /= n_texts
        cov_x_list[layer] /= n_texts  # i dont think this matters but its here anyway
        for h in range(n_heads):
            cov_q_list[layer][h] /= n_texts
            cov_k_list[layer][h] /= n_texts

    if logger:
        logger.info("Finished calibration and computed BI scores.")
    return cov_mlp_list, cov_q_list, cov_k_list, cov_x_list, bi_scores


def get_BI_score(x_in, x_out):
    Corr = []
    norm_diff = []
    x_in = x_in.double().to(device="cuda", dtype=torch.float64)
    x_out = x_out.double().to(device="cuda", dtype=torch.float64)

    normalizer = torch.norm(x_in, p=2, dim=2)

    diff = torch.norm((x_in - x_out), p=2, dim=2)
    diff = (diff / normalizer).mean()
    norm_diff.append(diff)

    x_in = normalize(x_in, p=2, dim=2)
    x_out = normalize(x_out, p=2, dim=2)

    corr = torch.diag((x_in @ x_out.mT)[0]).mean()
    Corr.append(corr)

    Corr = torch.tensor([x for x in Corr]).mean()
    norm_diff = torch.tensor([x for x in norm_diff]).mean()

    return (1 - Corr).item() * norm_diff.item()


@torch.no_grad()
def _make_fc_hook(layer_idx, cov_mlp_list, logger=None):
    def hook(module: torch.nn.Linear, inp, out):
        # out [B*T, D_int]
        # reallly we should grab the activation function directly
        # from the model its something like model.decoder....act_fn()
        act = torch.nn.functional.relu(out.to(dtype=torch.float64))
        H = act.detach().to(dtype=torch.float64).view(-1, act.size(-1))
        cov_mlp_list[layer_idx] += (H.T @ H).to(device="cpu")

    return hook


@torch.no_grad()
def _make_attn_hook(layer_idx, cov_q_list, cov_k_list, d_model, n_heads, head_dim, logger=None):
    def hook(module, inp, out):
        try:
            out = out.detach().to(dtype=torch.float64, device="cpu")
            q_block, k_block, _ = out.split(d_model, dim=2)
            Q = q_block.view(-1, d_model)
            K = k_block.view(-1, d_model)
            for h in range(n_heads):
                q_h = Q[:, h * head_dim : (h + 1) * head_dim]
                k_h = K[:, h * head_dim : (h + 1) * head_dim]
                cov_q_list[layer_idx][h] += q_h.T @ q_h
                cov_k_list[layer_idx][h] += k_h.T @ k_h
        except Exception as e:
            if logger:
                logger.warning(f"[Hook] Attn split failed at layer {layer_idx}: {e}")

    return hook


@torch.no_grad()
def _make_proj_hook(layer_idx, cov_list, n_heads, head_dim, d_model, logger=None):
    """
    For this to work on LLama Models we would have to use a nonlinear function on
        X @ W to create positional embeddings (RoPE for llama).

    in paper (sigma_r denotes positional embeddings -- not relevant for OPT)
    """

    def hook(module: torch.nn.Linear, inp, out):
        # try:
        # this can also be easily vectorized (no need to calculate C_proj over entire proj)
        proj_out = out.detach().to(dtype=torch.float64, device="cpu")  # [B,T, d_model]
        proj = proj_out.view(-1, d_model)  # [B*T, d_model]
        C_proj = proj.T @ proj
        C_proj = C_proj.to(device="cpu")
        for h in range(n_heads):
            h_proj = C_proj[h * head_dim : (h + 1) * head_dim, h * head_dim : (h + 1) * head_dim]
            cov_list[layer_idx][h] += h_proj

    # except Exception as e:
    #     if logger:
    #         logger.warning(f"[Hook] Q/K proj failed at layer {layer_idx}: {e}")

    return hook

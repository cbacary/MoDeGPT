import torch
from torch.nn.functional import cosine_similarity
from torch.types import Tensor


def calibrate_model(
    model: torch.nn.Module, tokenizer, texts, device="cpu", logger=None
):
    config = model.config
    n_layers = (
        getattr(config, "n_layer", None)
        or getattr(config, "num_hidden_layers", None)
        or getattr(config, "num_layers", None)
    )
    n_heads = getattr(config, "n_head", None) or getattr(
        config, "num_attention_heads", None
    )
    d_model = getattr(config, "hidden_size", None) or getattr(config, "dim", None)
    head_dim = d_model // n_heads

    if not getattr(model.config, "output_hidden_states", False):
        model.config.output_hidden_states = True

    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        transformer_blocks = model.transformer.h
        arch = "gpt"
    elif hasattr(model, "model") and hasattr(model.model, "decoder"):
        transformer_blocks = model.model.decoder.layers
        arch = "opt"
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        transformer_blocks = model.model.layers
        arch = "llama"
    else:
        raise RuntimeError("Unsupported model architecture")

    if logger:
        logger.info(f"Detected architecture: {arch}")

    def get_inner(transformer_block):
        # if arch == "gpt":
        #     n_inner = transformer_blocks.mlp.c_fc.out_features
        try:
            if arch == "opt":
                n_inner = transformer_blocks.fc1.out_features
            elif arch == "llama":
                n_inner = transformer_blocks.mlp.gate_proj.out_features
        except:
            n_inner = getattr(config, "intermediate_size", 4 * d_model)

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
    cov_x_list = [
        torch.zeros(d_model, d_model, dtype=torch.float64) for _ in range(n_layers)
    ]

    bi_scores = [0.0 for _ in range(n_layers)]
    bi_counts = [0 for _ in range(n_layers)]

    handles = []
    for i, block in enumerate(transformer_blocks):
        if arch == "gpt":
            handles.append(
                block.mlp.c_fc.register_forward_hook(
                    _make_fc_hook(i, cov_mlp_list, logger)
                )
            )
            handles.append(
                block.attn.c_attn.register_forward_hook(
                    _make_attn_hook(
                        i, cov_q_list, cov_k_list, d_model, n_heads, head_dim, logger
                    )
                )
            )
        elif arch == "opt":
            handles.append(
                block.fc1.register_forward_hook(_make_fc_hook(i, cov_mlp_list, logger))
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
        elif arch == "llama":
            handles.append(
                block.mlp.gate_proj.register_forward_hook(
                    _make_fc_hook(i, cov_mlp_list, logger)
                )
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

    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            for l in range(n_layers):
                x_in: Tensor = hidden_states[l].detach().to(torch.float64)  # [B, T, D]
                x_out = hidden_states[l + 1].detach().to(torch.float64)  # [B, T, D]

                x_in = x_in.view(-1, x_in.shape[-1])  # [B*T, D]
                x_out = x_out.view(-1, x_out.shape[-1])

                """
                 NOTE: This implemntation casts the correlation input to a float32
                       Appendix specifies that it should be held a float64 until 
                        later computations are performed.
                """
                x_in_cpu = x_in.to(device="cpu")
                cov_x_list[l] += x_in_cpu.T @ x_in_cpu

                # it was casted down before, and if it ain't broke don't fix it
                x_in = x_in.to(dtype=torch.float32)
                x_out = x_out.to(dtype=torch.float32)

                valid_mask = (x_in.norm(dim=1) > 0) & (x_out.norm(dim=1) > 0)
                if valid_mask.any():
                    cos_sim = (
                        cosine_similarity(x_in[valid_mask], x_out[valid_mask], dim=1)
                        .mean()
                        .item()
                    )
                    bi_scores[l] += 1.0 - cos_sim
                else:
                    bi_scores[l] += 1.0

                bi_counts[l] += 1

    for h in handles:
        h.remove()

    for i in range(n_layers):
        count = bi_counts[i]
        if count > 0:
            bi_scores[i] /= count
            cov_mlp_list[i] /= count
            for h in range(n_heads):
                cov_q_list[i][h] /= count
                cov_k_list[i][h] /= count

    if logger:
        logger.info("Finished calibration and computed BI scores.")
    return cov_mlp_list, cov_q_list, cov_k_list, cov_x_list, bi_scores


def _make_fc_hook(layer_idx, cov_mlp_list, logger=None):
    def hook(module, inp, out):
        # try:
        act = torch.nn.functional.gelu(
            out.to(dtype=torch.float64, device="cpu")
        )  # GELU activation function
        H = act.detach().to(dtype=torch.float64, device="cpu").view(-1, act.size(-1))
        cov_mlp_list[layer_idx] += H.T @ H

    # except Exception as e:
    #     if logger:
    #         logger.warning(f"[Hook] FC at layer {layer_idx} failed: {e}")

    return hook


def _make_attn_hook(
    layer_idx, cov_q_list, cov_k_list, d_model, n_heads, head_dim, logger=None
):
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


def _make_proj_hook(layer_idx, cov_list, n_heads, head_dim, logger=None):
    def hook(module, inp, out):
        try:
            proj_out = out.detach().to(dtype=torch.float64, device="cpu")
            for h in range(n_heads):
                h_proj = (
                    proj_out[:, :, h * head_dim : (h + 1) * head_dim]
                    .contiguous()
                    .view(-1, head_dim)
                )
                cov_list[layer_idx][h] += h_proj.T @ h_proj
        except Exception as e:
            if logger:
                logger.warning(f"[Hook] Q/K proj failed at layer {layer_idx}: {e}")

    return hook

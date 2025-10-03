import logging
import types

import torch
from torch.types import Tensor

from patchers.opt_patch import patched_forward

logger = logging.getLogger("MoDeGPT")


@torch.no_grad
def sqrt_M(M: Tensor, ridge_lambda=1e-4) -> Tensor:
    """
    Warning: Be weary of input being of precision torch.float64, may be because ridge lambda was too low,
    but this broke the result in some cases.
    The following is the code I originally used for sqrt_M. Its MUCH slower.
    However, note the following marginal ppl differerence against opt-1.3b WikiText2:
    Perplexity with eigenvalues:  16.44
    Perplexity with below method: 16.35

        # _C = cov[layer].to(device="cpu")
        # C = _C + torch.eye(_C.shape[0], device="cpu") * ridge_lambda

        # # === This is the most computationally exspensive task ======
        # # Can possible make cov_x a numpy array from the beginning to reduce this
        # C_np = C.detach().cpu().numpy()  # should already be on cpu but just in case
        # sqrt_C_np = sqrtm(C_np).real
        # sqrt_C = torch.from_numpy(sqrt_C_np).to(dtype=torch.float32, device="cuda")
    """

    M_reg = M + torch.eye(M.shape[0], device=M.device) * ridge_lambda
    eigenvalues, eigenvectors = torch.linalg.eigh(M_reg)
    sqrt_eigenvalues = torch.sqrt(eigenvalues.clamp(min=0))
    sqrt_M: Tensor = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T

    return sqrt_M.to(dtype=M.dtype)


@torch.no_grad
def slice_QK_dims(
    model,
    layer_idx: int,
    new_heads_Q: list[torch.Tensor],
    new_heads_K: list[torch.Tensor],
    new_bias_Q: list[torch.Tensor],
    new_bias_K: list[torch.Tensor],
    bias: bool,
):
    block = get_layer_block(model, layer_idx)
    self_attn = block.self_attn

    original_q = self_attn.q_proj
    original_k = self_attn.k_proj

    # not sure what dims shold be here
    Q_heads = torch.cat(new_heads_Q, dim=0).to(device="cuda", dtype=torch.float16)
    K_heads = torch.cat(new_heads_K, dim=0).to(device="cuda", dtype=torch.float16)
    bias_Q = torch.cat(new_bias_Q, dim=0).to(device="cuda", dtype=torch.float16)
    bias_K = torch.cat(new_bias_K, dim=0).to(device="cuda", dtype=torch.float16)

    new_layer_Q = torch.nn.Linear(
        in_features=Q_heads.shape[1],
        out_features=Q_heads.shape[0],
        device="cuda",
        dtype=torch.float16,
        bias=True if original_q.bias is not None and bias else False,
        # bias=False,
    )
    new_layer_Q.weight.data.copy_(Q_heads)
    if original_q.bias is not None and bias:
        new_layer_Q.bias.data.copy_(bias_Q)

    new_layer_K = torch.nn.Linear(
        in_features=K_heads.shape[1],
        out_features=K_heads.shape[0],
        device="cuda",
        dtype=torch.float16,
        bias=True if original_k.bias is not None and bias else False,
        # bias=False,
    )
    new_layer_K.weight.data.copy_(K_heads)
    if original_k.bias is not None and bias:
        new_layer_K.bias.data.copy_(bias_K)

    self_attn.k_proj = new_layer_K
    self_attn.q_proj = new_layer_Q

    self_attn.embed_dim = self_attn.k_proj.out_features


@torch.no_grad
def slice_VO_dims(
    model,
    layer_idx: int,
    new_heads_V: list[torch.Tensor],
    new_heads_O: list[torch.Tensor],
    bias: bool,
):
    block = get_layer_block(model, layer_idx)
    self_attn = block.self_attn

    if hasattr(self_attn, "out_proj"):
        original_o = self_attn.out_proj
    else:
        original_o = self_attn.o_proj

    original_v = self_attn.v_proj

    V_heads = torch.cat(new_heads_V, dim=0).to(device="cuda", dtype=torch.float16)
    O_heads = torch.cat(new_heads_O, dim=1).to(device="cuda", dtype=torch.float16)

    new_layer_V = torch.nn.Linear(
        in_features=V_heads.shape[1],
        out_features=V_heads.shape[0],
        device="cuda",
        dtype=torch.float16,
        # bias=True if original_o.bias is not None else False,
        bias=False,
    )
    new_layer_V.weight.data.copy_(V_heads)
    # output_features of o_proj is altered so bias is unusable
    if original_v.bias is not None and bias:
        new_layer_V.bias.data.copy_(original_v.bias.data)

    new_layer_O = torch.nn.Linear(
        in_features=O_heads.shape[1],
        out_features=O_heads.shape[0],
        device="cuda",
        dtype=torch.float16,
        bias=False,
        # bias=True if original_o.bias is not None else False,
    )
    new_layer_O.weight.data.copy_(O_heads)
    # if original_o.bias is not None and bias:
    #     new_layer_O.bias.data.copy_(original_o.bias.data)

    if hasattr(self_attn, "out_proj"):
        self_attn.out_proj = new_layer_O
    else:
        self_attn.o_proj = new_layer_O

    self_attn.v_proj = new_layer_V

    self_attn.embed_dim = self_attn.v_proj.out_features


def slice_gate_dims(model, layer_idx: int, up_weights: Tensor, down_weights: Tensor, bias: bool):
    block, up, down, arch = get_gate_projs(model, layer_idx=layer_idx)

    new_layer_U = torch.nn.Linear(
        in_features=up_weights.shape[1],
        out_features=up_weights.shape[0],
        device="cuda",
        dtype=torch.float16,
        # bias=True if up.bias is not None else False and bias,
        bias=False,
    )
    new_layer_U.weight.data.copy_(up_weights.to(torch.float16))
    # output_features of o_proj is altered so bias is unusable
    # if up.bias is not None and bias:
    #     new_layer_U.bias.data.copy_(up.bias.data)

    new_layer_D = torch.nn.Linear(
        in_features=down_weights.shape[1],
        out_features=down_weights.shape[0],
        device="cuda",
        dtype=torch.float16,
        bias=True if down.bias is not None and bias else False,
        # bias=False,
    )
    new_layer_D.weight.data.copy_(down_weights.to(torch.float16))
    if down.bias is not None and bias:
        new_layer_D.bias.data.copy_(down.bias.data)

    if arch == "OPT":
        block.fc1 = new_layer_U
        block.fc2 = new_layer_D
    elif arch == "GPT":
        block.mlp.c_fc = new_layer_U
        block.mlp.c_proj = new_layer_D
    elif arch == "LLAMA":
        block.mlp.gate_proj = new_layer_U
        block.mlp.down_proj = new_layer_D


def patch_forward_OPT(model):
    config = model.config
    n_layers = (
        getattr(config, "n_layer", None)
        or getattr(config, "num_hidden_layers", None)
        or getattr(config, "num_layers", None)
    )
    for layer in range(n_layers):
        block = get_layer_block(model, layer)
        self_attn = block.self_attn

        self_attn.forward = types.MethodType(patched_forward, self_attn)


def get_gate_projs(model, layer_idx):
    try:
        block = model.model.decoder.layers[layer_idx]  # OPT
        up = block.fc1  # [D_int, D_h]
        down = block.fc2  # [D_h, D_int]
        return block, up, down, "OPT"
    except AttributeError:
        try:
            block = model.transformer.h[layer_idx]  # GPT
            up = block.mlp.c_fc
            down = block.mlp.c_proj

            return block, up, down, "GPT"
        except AttributeError:
            block = model.model.layers[layer_idx]  # LLaMA
            up = block.mlp.gate_proj
            down = block.mlp.down_proj
            return block, up, down, "LLAMA"


def get_decoder(model, layer_idx):
    if hasattr(model.model, "decoder"):
        return model.model.decoder
    # elif for other models, dont know what they got
    else:
        raise AttributeError


def get_layer_block(model, layer_idx):
    if hasattr(model.model, "decoder"):
        block = model.model.decoder.layers[layer_idx]
    elif hasattr(model.model, "layers"):
        block = model.model.layers[layer_idx]
    else:
        raise AttributeError

    return block


def get_embedders(model):
    decoder = model.model.decoder
    return decoder.embed_tokens, decoder.embed_positions


def get_Q_K_weights(model, layer_idx):
    if hasattr(model.model, "decoder"):
        block = model.model.decoder.layers[layer_idx]
    elif hasattr(model.model, "layers"):
        block = model.model.layers[layer_idx]
    else:
        raise AttributeError

    W_q = block.self_attn.q_proj.weight
    W_k = block.self_attn.k_proj.weight
    return W_q, W_k


def get_V_O_weights(model, layer_idx):
    if hasattr(model.model, "decoder"):
        block = model.model.decoder.layers[layer_idx]
        W_v = block.self_attn.v_proj.weight
        W_o = block.self_attn.out_proj.weight
    elif hasattr(model.model, "layers"):
        block = model.model.layers[layer_idx]
        W_v = block.self_attn.v_proj.weight
        W_o = block.self_attn.o_proj.weight
    else:
        raise AttributeError

    return W_v, W_o


def allocate_global_sparsity(
    bi_scores: list[float],
    compression_ratio: float,
    smoothing: float = 1.0,
):
    from torch.nn.functional import softmax

    n_layers = len(bi_scores)

    logger.info(
        f"Allocating global sparsity using compression_ratio {compression_ratio:.4f} and temperature {smoothing}"
    )

    L = n_layers
    phi_avg = compression_ratio
    epsilon = smoothing

    s = torch.tensor(bi_scores).to(torch.float64)
    phi = L * phi_avg * softmax(-s / epsilon, dim=0)
    phi[0] = 0.0
    for count, (bi_score, sparsity) in enumerate(zip(bi_scores, phi)):
        logger.info(f"Layer {count}: sparisty = {sparsity:.4f} (BI = {bi_score:.4f})")

    return 1 - phi


# def allocate_global_sparsity(
#     bi_scores,
#     model_config,
#     compression_ratio: float,
#     temperature: float = 1.0,
#     min_keep: float = 0.05,
#     max_keep: float = 0.95,
# ):
#     """
#     Allocate layer-wise retention ratio φ_i based on BI scores.

#     Implements softmax allocation with optional min/max constraints and normalization.

#     Layer 0 is skipped from compression and assigned keep_ratio = 1.0.

#     Args:
#         bi_scores (List[float]): Per-layer BI scores
#         model_config: HF model config
#         target_keep_ratio (float): Global average keep ratio (e.g., 0.5)
#         temperature (float): Softmax temperature
#         min_keep (float): Minimum per-layer keep ratio
#         max_keep (float): Maximum per-layer keep ratio

#     Returns:
#         List[float]: Per-layer keep ratios φ_i
#     """
#     target_keep_ratio = 1 - compression_ratio
#     import math

#     n_layers = len(bi_scores)
#     expected_layers = (
#         getattr(model_config, "num_hidden_layers", None)
#         or getattr(model_config, "n_layer", None)
#         or getattr(model_config, "num_layers", None)
#     )
#     if expected_layers is None or n_layers != expected_layers:
#         raise ValueError(
#             f"Mismatch between BI scores ({n_layers}) and model layers ({expected_layers})"
#         )

#     logger.info(
#         f"Allocating global sparsity using target keep ratio {target_keep_ratio:.4f} and temperature {temperature}"
#     )

#     # === Step 1: Fix Layer 0 ===
#     keep_ratios = [1.0]  # Layer 0 is always kept

#     # === Step 2: Compute softmax for layers 1 to n-1 ===
#     bi_sub = bi_scores[1:]
#     max_score = max(bi_sub)
#     shifted = [s - max_score for s in bi_sub]  # numerical stability
#     raw_importance = [math.exp(-s / temperature) for s in shifted]
#     total = sum(raw_importance)
#     normalized = [x / total for x in raw_importance]

#     # === Step 3: Allocate keep_ratios for remaining layers ===
#     remaining_budget = (
#         target_keep_ratio * n_layers - 1.0
#     )  # subtract Layer 0's fixed 1.0
#     initial = [w * remaining_budget for w in normalized]
#     clipped = [min(max(r, min_keep), max_keep) for r in initial]

#     # === Step 4: Renormalize clipped layers if needed ===
#     clipped_sum = sum(clipped)
#     if not math.isclose(clipped_sum, remaining_budget, rel_tol=1e-3):
#         scale = remaining_budget / clipped_sum
#         clipped = [min(max(r * scale, min_keep), max_keep) for r in clipped]

#     # === Step 5: Combine final keep_ratios ===
#     keep_ratios.extend(clipped)

#     # === Step 6: Logging ===
#     for i, (score, ratio) in enumerate(zip(bi_scores, keep_ratios)):
#         logger.info(f"Layer {i:2d}: keep_ratio = {ratio:.4f} (BI = {score:.4f})")

#     return keep_ratios

import logging
import types

import torch
from torch.types import Tensor


logger = logging.getLogger("MoDeGPT")

from model_utils import dtype_p


@torch.no_grad
def sqrt_M(M: Tensor, ridge_lambda=1e-4) -> Tensor:
    """
    Warning: Be weary of input being of precision dtype_p, may be because ridge lambda was too low,
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
    if len(new_bias_Q):
        bias_Q = torch.cat(new_bias_Q, dim=0).to(device="cuda", dtype=torch.float16)
    if len(new_bias_K):
        bias_K = torch.cat(new_bias_K, dim=0).to(device="cuda", dtype=torch.float16)

    new_layer_Q = torch.nn.Linear(
        in_features=Q_heads.shape[1],
        out_features=Q_heads.shape[0],
        device="cuda",
        dtype=torch.float16,
        bias=original_q.bias is not None and bias and len(new_bias_Q),
        # bias=False,
    )
    new_layer_Q.weight.data.copy_(Q_heads)
    if new_layer_Q.bias is not None:
        new_layer_Q.bias.data.copy_(bias_Q)

    new_layer_K = torch.nn.Linear(
        in_features=K_heads.shape[1],
        out_features=K_heads.shape[0],
        device="cuda",
        dtype=torch.float16,
        bias=original_k.bias is not None and bias and len(new_bias_K),
        # bias=False,
    )
    new_layer_K.weight.data.copy_(K_heads)
    if new_layer_K.bias is not None:
        new_layer_K.bias.data.copy_(bias_K)

    self_attn.k_proj = new_layer_K
    self_attn.q_proj = new_layer_Q


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
    # if original_v.bias is not None and bias:
    #     new_layer_V.bias.data.copy_(original_v.bias.data)

    new_layer_O = torch.nn.Linear(
        in_features=O_heads.shape[1],
        out_features=O_heads.shape[0],
        device="cuda",
        dtype=torch.float16,
        # bias=False,
        bias=True if original_o.bias is not None and bias else False,
    )
    new_layer_O.weight.data.copy_(O_heads)
    if original_o.bias is not None and bias:
        new_layer_O.bias.data.copy_(original_o.bias.data)

    if hasattr(self_attn, "out_proj"):
        self_attn.out_proj = new_layer_O
    else:
        self_attn.o_proj = new_layer_O

    self_attn.v_proj = new_layer_V

    # self_attn.embed_dim = self_attn.v_proj.out_features


def slice_gate_dims(
    model,
    layer_idx: int,
    up_weights: Tensor,
    gate_weights: Tensor | None,
    down_weights: Tensor,
    new_bias_g: Tensor | None,
    new_bias_u: Tensor,
    bias: bool,
):
    """
    gate_weights, new_bias_g only used for llama
    """
    block, up, down, gate, arch = get_gate_projs(model, layer_idx=layer_idx)

    new_layer_U = torch.nn.Linear(
        in_features=up_weights.shape[1],
        out_features=up_weights.shape[0],
        device="cuda",
        dtype=torch.float16,
        bias=True if up.bias is not None and bias else False,
        # bias=False,
    )
    new_layer_U.weight.data.copy_(up_weights.to(torch.float16))
    if bias and new_bias_u is not None:
        logger.info("Copying up layer bias")
        new_layer_U.bias.data.copy_(new_bias_u)

    if arch == "llama":
        new_layer_G = torch.nn.Linear(
            in_features=gate_weights.shape[1],
            out_features=gate_weights.shape[0],
            device="cuda",
            dtype=torch.float16,
            bias=True if gate.bias is not None and bias else False,
            # bias=False,
        )
        new_layer_G.weight.data.copy_(gate_weights.to(torch.float16))
        if bias and new_bias_g is not None:
            logger.info("Copying gate layer bias")
            new_layer_G.bias.data.copy_(new_bias_g)

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
        logger.info("Copying down layer bias")
        new_layer_D.bias.data.copy_(down.bias.data)

    if arch == "opt":
        block.fc1 = new_layer_U
        block.fc2 = new_layer_D
    elif arch == "gpt":
        block.mlp.c_fc = new_layer_U
        block.mlp.c_proj = new_layer_D
    elif arch == "llama":
        block.mlp.up_proj = new_layer_U
        block.mlp.gate_proj = new_layer_G
        block.mlp.down_proj = new_layer_D


def get_gate_projs(model, layer_idx):
    try:
        block = model.model.decoder.layers[layer_idx]  # OPT
        up = block.fc1  # [D_int, D_h]
        down = block.fc2  # [D_h, D_int]
        return block, up, down, None, "opt"
    except AttributeError:
        try:
            block = model.transformer.h[layer_idx]  # GPT
            up = block.mlp.c_fc
            down = block.mlp.c_proj

            return block, up, down, None, "gpt"
        except AttributeError:
            block = model.model.layers[layer_idx]  # LLaMA
            up = block.mlp.up_proj
            down = block.mlp.down_proj
            gate = block.mlp.gate_proj
            return block, up, down, gate, "llama"


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

    s = torch.tensor(bi_scores).to(dtype_p)
    phi = L * phi_avg * softmax(-s / epsilon, dim=0)
    # phi[0] = 0.0
    for count, (bi_score, sparsity) in enumerate(zip(bi_scores, phi)):
        logger.info(f"Layer {count}: sparisty = {sparsity:.4f} (BI = {bi_score:.4f})")

    return 1 - phi

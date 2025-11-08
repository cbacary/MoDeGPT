import logging

import torch

from calibration import get_model_attrs

# from transformers.models.opt.modeling_opt import OPTDecoderLayer
from compression_utils import get_Q_K_weights, get_V_O_weights, get_gate_projs

logger = logging.getLogger("MoDeGPT")


def patch_config(model: torch.nn.Module):
    config = model.config

    n_layers, n_heads, d_model, head_dim, arch = get_model_attrs(model)

    # handle qk/vo separately in case only one stage is used
    qk_ranks = []
    vo_ranks = []
    gate_ranks = []
    for layer in range(n_layers):
        query_weight, _ = get_Q_K_weights(model, layer)
        value_weight, _ = get_V_O_weights(model, layer)
        _, up_proj, _, _, _ = get_gate_projs(model, layer)

        q_rank = query_weight.shape[0]  # q/k
        v_rank = value_weight.shape[0]  # v/o
        gate_rank = up_proj.weight.shape[0]  # u/d

        qk_ranks.append(q_rank)
        vo_ranks.append(v_rank)
        gate_ranks.append(gate_rank)

    # just break these values so we know where's something's going wrong later
    config.ffn_dim = -1
    config.qk_ranks = qk_ranks
    config.vo_ranks = vo_ranks
    config.gate_ranks = gate_ranks
    if arch == "opt":
        config.auto_map = {"AutoModelForCausalLM": "OPTRebuild.OPTForCausalLM"}
    if arch == "llama":
        config.auto_map = {"AutoModelForCausalLM": "LlamaRebuild.LlamaForCausalLM"}

import logging

import torch

# from transformers.models.opt.modeling_opt import OPTDecoderLayer
from compression_utils import get_Q_K_weights, get_V_O_weights, get_gate_projs
import copy

logger = logging.getLogger("MoDeGPT")


def patch_config(model: torch.nn.Module):
    og_config = copy.deepcopy(model.config)
    config = model.config

    n_layers, arch = config.num_hidden_layers, config.model_type

    # handle qk/vo separately in case only one stage is used
    q_ranks = []
    k_ranks = []
    v_ranks = []
    o_ranks = []
    gate_ranks = []
    for layer in range(n_layers):
        query_weight, key_weight = get_Q_K_weights(model, layer)
        value_weight, output_weight = get_V_O_weights(model, layer)
        _, up_proj, _, _, _ = get_gate_projs(model, layer)

        q_rank = query_weight.shape[0]  # q/k
        k_rank = key_weight.shape[0]  # q/k

        v_rank = value_weight.shape[0]  # v/o
        o_rank = output_weight.shape[1]
        gate_rank = up_proj.weight.shape[0]  # u/d

        q_ranks.append(q_rank)
        k_ranks.append(k_rank)
        v_ranks.append(v_rank)
        o_ranks.append(o_rank)
        gate_ranks.append(gate_rank)

    # just break these values so we know where's something's going wrong later
    config.ffn_dim = -1
    config.q_ranks = q_ranks
    config.k_ranks = k_ranks
    config.v_ranks = v_ranks
    config.o_ranks = o_ranks
    config.gate_ranks = gate_ranks
    if arch == "opt":
        config.auto_map = {"AutoModelForCausalLM": "OPTRebuild.OPTForCausalLM"}
    if arch == "llama":
        config.auto_map = {"AutoModelForCausalLM": "LlamaRebuild.LlamaForCausalLM"}

    return og_config

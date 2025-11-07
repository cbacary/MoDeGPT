import logging
from typing import Callable, Optional

import torch
from transformers import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.opt.modeling_opt import eager_attention_forward

from calibration import get_model_attrs

from transformers.models.opt.modeling_opt import OPTDecoderLayer

logger = logging.getLogger("MoDeGPT")


def patch_config(model: torch.nn.Module):
    config = model.config

    n_layers, n_heads, d_model, head_dim, arch = get_model_attrs(model)

    # handle qk/vo separately in case only one stage is used
    qk_ranks = []
    vo_ranks = []
    gate_ranks = []
    for layer in range(n_layers):
        block: OPTDecoderLayer = model.model.decoder.layers[layer]
        up_weight = block.fc1.weight
        query_weight = block.self_attn.q_proj.weight
        value_weight = block.self_attn.v_proj.weight

        q_rank = query_weight.shape[0]  # v/o
        v_rank = value_weight.shape[0]  # q/k
        gate_rank = up_weight.shape[0]  # u/d

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


#### NOTE TO SELF: see OPTDecoderLayer.final_layer_norm #####
@torch.no_grad()
def patched_forward(
    self,
    hidden_states: torch.Tensor,
    past_key_values: Optional[tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    cache_position: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
    """
    ::::Copied from transformers.models.opt.modeling_opt OPTAttention.forward::::
    Input shape: Batch x Time x Channel
    """

    bsz, tgt_len, _ = hidden_states.size()

    # Scaling is susceptible to floating point arithmetics' inprecisions
    # which can lead to different results (this is dependent from model
    # to model, e.g. whisper is one such case). We therefore keep the
    # original order of scaling to follow the original implementation
    # and enforce no scaling (1.0) in the attention call below.
    query_states = self.q_proj(hidden_states) * self.scaling
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    new_total_q_dim = query_states.shape[-1]
    new_q_head_dim = new_total_q_dim // self.num_heads
    new_total_k_dim = key_states.shape[-1]
    new_k_head_dim = new_total_k_dim // self.num_heads
    new_total_v_dim = value_states.shape[-1]
    new_v_head_dim = new_total_v_dim // self.num_heads

    print(f"q.shape = {query_states.shape} -- {new_q_head_dim}")
    print(f"k.shape = {key_states.shape} -- {new_k_head_dim}")
    print(f"v.shape = {value_states.shape} -- {new_v_head_dim}")
    print(f"self.embed_dim = {self.embed_dim}")
    print(f"hidden_states.size() = {hidden_states.size()}")

    query_states = query_states.view(bsz, -1, self.num_heads, new_q_head_dim).transpose(1, 2)

    key_states = key_states.view(bsz, -1, self.num_heads, new_k_head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, -1, self.num_heads, new_v_head_dim).transpose(1, 2)

    if past_key_values is not None:
        # save all key/value_states to cache to be re-used for fast auto-regressive generation
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, {"cache_position": cache_position}
        )

    attention_interface: Callable = eager_attention_forward

    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and output_attentions:
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.dropout,
        scaling=1.0,
        **kwargs,
    )

    print(f"attn_output.shape = {attn_output.shape}")
    attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
    print(f"reshaped_attn_output.shape = {attn_output.shape}")
    attn_output = self.out_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights

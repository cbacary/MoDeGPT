import logging
from typing import Callable, Optional

import torch
from transformers import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.opt.modeling_opt import eager_attention_forward

logger = logging.getLogger("MoDeGPT")


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
    new_head_dim = new_total_q_dim // self.num_heads
    new_total_k_dim = key_states.shape[-1]
    new_k_head_dim = new_total_k_dim // self.num_heads
    new_total_v_dim = value_states.shape[-1]
    new_v_head_dim = new_total_v_dim // self.num_heads

    query_states = query_states.view(bsz, -1, self.num_heads, new_head_dim).transpose(
        1, 2
    )

    key_states = key_states.view(bsz, -1, self.num_heads, new_k_head_dim).transpose(
        1, 2
    )
    value_states = value_states.view(bsz, -1, self.num_heads, new_v_head_dim).transpose(
        1, 2
    )

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
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

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

    attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
    attn_output = self.out_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights

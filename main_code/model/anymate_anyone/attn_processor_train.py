# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py



from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import numpy as np

import diffusers
from torch.nn.functional import scaled_dot_product_attention


class ReferenceAttnProcessorWithZeroConvolution:
    """
    Optimized reference attention processor with token pruning and global top-k value routing.
    """

    def __init__(self, hidden_size=None, cross_attention_dim=None, mode = "training"):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Requires PyTorch >= 2.0 for scaled dot product attention.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        external_kv=None,
        temb=None,
        queries_ratio = 1.0   # when inference, can given by dinov2 model
    ):
        # Input shape handling
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height*width).transpose(1, 2)
        
        # Process encoder states
        encoder_hidden_states = encoder_hidden_states.to(hidden_states.dtype)
        if encoder_hidden_states.ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height*width).transpose(1, 2)

        
        B = hidden_states.shape[0]
        # Attention projections
        query = attn.head_to_batch_dim(attn.to_q(hidden_states))
        key = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states))
        value = attn.head_to_batch_dim(attn.to_v(encoder_hidden_states))

        learnable_token = attn.learnable_token.expand(B, -1, -1)
        learnable_token = attn.head_to_batch_dim(learnable_token)   # B * h, l_t, C'

        # summarize reference
        attn_ref = torch.bmm(learnable_token, key.transpose(-1, -2)) * attn.scale
        attn_ref = torch.softmax(attn_ref, dim=-1)
        summarized_token = torch.bmm(attn_ref, key) # pay attention, here is key, because we want to summarize key from reference!

        # Q attends summarized_token
        attn_summary = torch.bmm(query, summarized_token.transpose(-1, -2)) * attn.scale
        attn_summary_gate = torch.sigmoid(attn_summary.mean(-1)) # b * heads, l_q, 1
        attn_summary_gate = attn.batch_to_head_dim(attn_summary_gate) # b, l_q, 1 * heads
        attn_summary_gate = attn_summary_gate.mean(-1)    # b, l_q, 1

        # ref-attention
        # attention_scores = torch.bmm(query, key.transpose(-1, -2)) * attn.scale
        # atten_vals = F.softmax(attention_scores, dim=-1)
        # weighted_values = torch.bmm(atten_vals, value) # full attention
        
        weighted_values = F.scaled_dot_product_attention(
            query, key, value, 
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # project back
        hidden_states_out = attn.to_out[0](attn.batch_to_head_dim(weighted_values))
        hidden_states_out = attn.to_out[1](hidden_states_out)
        hidden_states_out = hidden_states_out

        # Apply zero-initialized projection
        hidden_states_out = attn.zero_linear(hidden_states_out)
        hidden_states_out = hidden_states_out * attn_summary_gate.unsqueeze(-1)  # [B, L', C]
        
       
       # Final processing
        if input_ndim == 4:
            hidden_states_out = hidden_states_out.transpose(1, 2).view(batch_size, channel, height, width)
        return hidden_states_out

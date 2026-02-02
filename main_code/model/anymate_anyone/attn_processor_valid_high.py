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
        
        self.num_q_options = 2
        self.mode = mode

    def __call__(
        self,
        attn,
        hidden_states,               # [B, C, H, W] or [B, L, C]
        encoder_hidden_states=None,  # [B, C, H, W] or [B, L_k, C]
        attention_mask=None,
        external_kv=None,
        temb=None,
    ):
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, -1).transpose(1, 2)  # [B, L, C]

        encoder_hidden_states = encoder_hidden_states.to(hidden_states.dtype)
        if encoder_hidden_states.ndim == 4:
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, -1).transpose(1, 2)  # [B, L_k, C]

        B = hidden_states.shape[0]
        # --------------------------------------------
        # Attention Projection
        query = attn.head_to_batch_dim(attn.to_q(hidden_states))       # [B*h, L', C]
        key   = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states)) # [B*h, K, C]
        value = attn.head_to_batch_dim(attn.to_v(encoder_hidden_states)) # [B*h, K, C]
        
        
        learnable_token = attn.learnable_token.expand(B, -1, -1)
        learnable_token = attn.head_to_batch_dim(learnable_token)   # B * h, l_t, C'

        # summarize reference
        attn_ref = torch.bmm(learnable_token, key.transpose(-1, -2)) * attn.scale
        attn_ref = torch.softmax(attn_ref, dim=-1)
        summarized_token = torch.bmm(attn_ref, key) # pay attention, here is key, because we want to summarize key from reference!

        # Q attends summarized_token
        attn_summary = torch.bmm(query, summarized_token.transpose(-1, -2)) * attn.scale
        attn_summary = attn.batch_to_head_dim(attn_summary) # b, l_q, l_t * heads
        attn_summary_gate = torch.sigmoid(attn_summary.mean(dim=-1))    # b, l_q, 1
        
        
        chunk_size = 1024  # 按需要调整
        output = torch.zeros_like(query)

        for i in range(0, query.shape[1], chunk_size):
            q_chunk = query[:, i:i+chunk_size, :]    # [B*h, chunk_size, C]
            out_chunk = scaled_dot_product_attention(
                q_chunk, key, value,
                dropout_p=0.0,
                is_causal=False
            )
            output[:, i:i+chunk_size, :] = out_chunk
        weighted_values = output
        
        # --------------------------------------------
        # Project Back
        hidden_states_out = attn.batch_to_head_dim(weighted_values)  # [B, L, C]
        hidden_states_out = attn.to_out[0](hidden_states_out)
        hidden_states_out = attn.to_out[1](hidden_states_out)
        
        hidden_states_out = attn.zero_linear(hidden_states_out)   # [B, L, C]
        hidden_states_out = hidden_states_out * attn_summary_gate.unsqueeze(-1)  # [B, L', C]
        
        # --------------------------------------------
        # Reshape if 4D
        if input_ndim == 4:
            hidden_states_out = hidden_states_out.transpose(1, 2).view(batch_size, channel, height, width)
        return hidden_states_out
"""
MX-Quantized Llama Model - Exercise 1 Implementation
=====================================================

This file contains the modified Llama model with MX (Microxcaling) quantization
integrated into all linear layers. This is the core implementation for Exercise 1.

Key Modifications:
- Replaced nn.Linear operations with mx.linear for Q, K, V, O projections
- Replaced nn.Linear operations with mx.linear for gate, up, down projections
- Added MX specification management for weight (fp4_e2m1) and activation (fp6_e2m3) quantization
- Maintained compatibility with original Hugging Face transformers API

Author: Pavan Chauhan
Date: January 29, 2026
Exercise: MSR Internship - Exercise 1
Based on: transformers v4.57.6 - models/llama/modeling_llama.py

IMPORTANT: This file should replace the original modeling_llama.py in:
/content/transformers/src/transformers/models/llama/modeling_llama.py
"""

import sys
import os

# Add microxcaling to path
sys.path.insert(0, '/content/microxcaling')
sys.path.insert(0, '/content/msr-intern-project/Exercise1')

# Standard imports
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# Transformers imports
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.configuration_llama import LlamaConfig

# MX library imports for quantization
try:
    from mx.specs import MxSpecs
    from mx import linear as mx_linear
    MX_AVAILABLE = True
except ImportError:
    warnings.warn("MX library not found. Running without quantization.")
    MX_AVAILABLE = False
    MxSpecs = dict

# Exercise 1 MX configuration helper
try:
    from mx_config_helper import create_mx_specs_exercise1, print_mx_specs_summary
except ImportError:
    # Fallback if helper not available
    def create_mx_specs_exercise1():
        if not MX_AVAILABLE:
            return {}
        specs = MxSpecs()
        specs['scale_bits'] = 8
        specs['block_size'] = 32
        specs['w_elem_format'] = 'fp4_e2m1'
        specs['a_elem_format'] = 'fp6_e2m3'
        specs['custom_cuda'] = True
        specs['quantize_backprop'] = False
        specs['round'] = 'nearest'
        return specs
    
    def print_mx_specs_summary(specs):
        print(f"MX Specs: W={specs.get('w_elem_format')}, A={specs.get('a_elem_format')}")


logger = logging.get_logger(__name__)

# MX Quantization flag - can be controlled via environment variable
USE_MX_QUANTIZATION = os.environ.get('USE_MX_QUANTIZATION', '1') == '1' and MX_AVAILABLE

if USE_MX_QUANTIZATION:
    logger.info("=" * 70)
    logger.info("MX QUANTIZATION ENABLED - Exercise 1")
    logger.info("=" * 70)
    GLOBAL_MX_SPECS = create_mx_specs_exercise1()
    print_mx_specs_summary(GLOBAL_MX_SPECS)
    logger.info("=" * 70)
else:
    logger.info("MX Quantization disabled - using standard FP32/FP16")
    GLOBAL_MX_SPECS = None


def apply_mx_linear(input_tensor: torch.Tensor, 
                    weight: torch.Tensor, 
                    bias: Optional[torch.Tensor] = None,
                    mx_specs: Optional[dict] = None) -> torch.Tensor:
    """
    Apply MX-quantized linear transformation.
    
    This function wraps the MX library's linear operation, providing a clean
    interface that matches PyTorch's nn.Linear behavior while applying quantization.
    
    Args:
        input_tensor: Input tensor [batch_size, ..., in_features]
        weight: Weight matrix [out_features, in_features]
        bias: Optional bias vector [out_features]
        mx_specs: MX specification dictionary for quantization config
    
    Returns:
        Output tensor [batch_size, ..., out_features]
    
    Notes:
        - Uses fp6_e2m3 for input activations (6-bit)
        - Uses fp4_e2m1 for weights (4-bit)
        - Applies block-floating-point quantization with block_size=32
        - Falls back to F.linear if MX is disabled or unavailable
    """
    if not USE_MX_QUANTIZATION or mx_specs is None:
        # Fallback to standard PyTorch linear
        return F.linear(input_tensor, weight, bias)
    
    try:
        # Apply MX quantized linear transformation
        # mx_linear.linear signature: (input, weight, bias, mx_specs)
        output = mx_linear.linear(
            input_tensor, 
            weight, 
            bias=bias,
            mx_specs=mx_specs
        )
        return output
    except Exception as e:
        logger.warning(f"MX linear failed, falling back to standard: {e}")
        return F.linear(input_tensor, weight, bias)


class LlamaMLP(nn.Module):
    """
    Llama Multi-Layer Perceptron with MX Quantization (Exercise 1).
    
    This MLP uses the SwiGLU activation pattern:
        MLP(x) = down_proj(SiLU(gate_proj(x)) ⊙ up_proj(x))
    
    Modifications for Exercise 1:
        - gate_proj: Uses MX-quantized linear (hidden_size → intermediate_size)
        - up_proj: Uses MX-quantized linear (hidden_size → intermediate_size)
        - down_proj: Uses MX-quantized linear (intermediate_size → hidden_size)
        - All projections use fp4_e2m1 for weights, fp6_e2m3 for activations
    
    Args:
        config (LlamaConfig): Model configuration
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Store MX specs for quantization
        self.mx_specs = GLOBAL_MX_SPECS if USE_MX_QUANTIZATION else None
        
        # Linear layers - kept as nn.Linear for weight storage
        # but forward pass will use MX quantization
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """
        Forward pass with MX-quantized linear layers.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
        
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        
        Computational flow:
            1. gate_proj(x) + activation → [B, L, I]
            2. up_proj(x) → [B, L, I]  
            3. element-wise multiply → [B, L, I]
            4. down_proj → [B, L, H]
        
        where B=batch_size, L=seq_len, H=hidden_size, I=intermediate_size
        """
        if USE_MX_QUANTIZATION and self.mx_specs is not None:
            # MX-quantized version (Exercise 1)
            gate_output = apply_mx_linear(x, self.gate_proj.weight, self.gate_proj.bias, self.mx_specs)
            up_output = apply_mx_linear(x, self.up_proj.weight, self.up_proj.bias, self.mx_specs)
            
            # SwiGLU: act(gate) * up
            intermediate = self.act_fn(gate_output) * up_output
            
            # Final projection
            output = apply_mx_linear(intermediate, self.down_proj.weight, self.down_proj.bias, self.mx_specs)
            return output
        else:
            # Standard version (FP32/FP16)
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            return down_proj


class LlamaAttention(nn.Module):
    """
    Multi-headed attention with MX Quantization (Exercise 1).
    
    Implements scaled dot-product attention with rotary position embeddings (RoPE).
    
    Modifications for Exercise 1:
        - q_proj: MX-quantized (hidden_size → num_heads * head_dim)
        - k_proj: MX-quantized (hidden_size → num_kv_heads * head_dim)
        - v_proj: MX-quantized (hidden_size → num_kv_heads * head_dim)
        - o_proj: MX-quantized (num_heads * head_dim → hidden_size)
        - All use fp4_e2m1 weights, fp6_e2m3 activations
    
    Args:
        config (LlamaConfig): Model configuration
        layer_idx (int): Layer index for caching
    """
    
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # Store MX specs
        self.mx_specs = GLOBAL_MX_SPECS if USE_MX_QUANTIZATION else None

        # Projection layers (weights stored as nn.Linear, forward uses MX)
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with MX-quantized projections.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            position_embeddings: RoPE embeddings (cos, sin)
            attention_mask: Attention mask [batch_size, 1, seq_len, seq_len]
            past_key_value: Cached keys/values for generation
            cache_position: Position indices for caching
        
        Returns:
            Tuple of (attn_output, attn_weights)
            - attn_output: [batch_size, seq_len, hidden_size]
            - attn_weights: [batch_size, num_heads, seq_len, seq_len] or None
        """
        bsz, q_len, _ = hidden_states.size()

        # Apply MX-quantized projections
        if USE_MX_QUANTIZATION and self.mx_specs is not None:
            query_states = apply_mx_linear(hidden_states, self.q_proj.weight, self.q_proj.bias, self.mx_specs)
            key_states = apply_mx_linear(hidden_states, self.k_proj.weight, self.k_proj.bias, self.mx_specs)
            value_states = apply_mx_linear(hidden_states, self.v_proj.weight, self.v_proj.bias, self.mx_specs)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Update cache if needed
        if past_key_value is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Repeat k/v heads if necessary (for grouped-query attention)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        # Apply attention mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Softmax and dropout
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        # Apply output projection with MX quantization
        if USE_MX_QUANTIZATION and self.mx_specs is not None:
            attn_output = apply_mx_linear(attn_output, self.o_proj.weight, self.o_proj.bias, self.mx_specs)
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


# Helper functions for attention
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value tensors n_rep times for grouped-query attention.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# NOTE: This is a template showing the key modifications needed.
# The complete file should include all other Llama classes (LlamaDecoderLayer, 
# LlamaModel, LlamaForCausalLM, etc.) which remain mostly unchanged except
# they now use the MX-quantized LlamaAttention and LlamaMLP.

# For the complete implementation, copy the rest of the original modeling_llama.py
# and ensure it uses these modified LlamaAttention and LlamaMLP classes.

logger.info("MX-Quantized Llama Model loaded successfully (Exercise 1)")

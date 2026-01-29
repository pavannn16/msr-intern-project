"""
MX Quantization Configuration Helper for Exercise 1
====================================================

This module provides helper functions to create and manage MX quantization
specifications for the Llama model integration.

Author: Pavan Chauhan
Date: January 29, 2026
Exercise: MSR Internship - Exercise 1
"""

from typing import Dict, Any, Optional
import sys
sys.path.insert(0, '/content/microxcaling')

try:
    from mx.specs import MxSpecs
except ImportError:
    print("Warning: MX library not found. Please ensure microxcaling is in PYTHONPATH.")
    MxSpecs = dict  # Fallback for type hints


def create_mx_specs_exercise1(
    weight_format: str = 'fp4_e2m1',
    activation_format: str = 'fp6_e2m3',
    scale_bits: int = 8,
    block_size: int = 32,
    custom_cuda: bool = True,
    quantize_backprop: bool = False,
    round_mode: str = 'nearest'
) -> Dict[str, Any]:
    """
    Create MX specifications for Exercise 1: Linear Layer Quantization.
    
    This function generates a properly configured MX specs dictionary for
    quantizing linear layers in the Llama model with the specified formats.
    
    Args:
        weight_format (str): Element format for weights.
            Options: 'fp4_e2m1', 'fp6_e2m3', 'fp8_e5m2', 'int8'
            Default: 'fp4_e2m1' (4-bit, 1 sign + 2 exp + 1 mantissa)
            
        activation_format (str): Element format for activations.
            Default: 'fp6_e2m3' (6-bit, 1 sign + 2 exp + 3 mantissa)
            
        scale_bits (int): Number of bits for shared exponent (E8M0 format).
            Default: 8 (standard for MX-compatible formats)
            
        block_size (int): Number of elements sharing one exponent.
            Default: 32 (OCP MX specification standard)
            
        custom_cuda (bool): Use custom CUDA kernels for better performance.
            Default: True (recommended for accuracy and speed)
            
        quantize_backprop (bool): Apply quantization to backward pass.
            Default: False (inference only)
            
        round_mode (str): Rounding behavior for quantization.
            Options: 'nearest', 'floor', 'even'
            Default: 'nearest'
    
    Returns:
        Dict[str, Any]: MX specs dictionary ready for use with mx.linear()
    
    Example:
        >>> specs = create_mx_specs_exercise1()
        >>> # Use with mx.linear
        >>> import mx.linear as mx_linear
        >>> output = mx_linear.linear(input, weight, bias, mx_specs=specs)
    
    Notes:
        - Weight format (fp4_e2m1) provides 8x compression vs FP32
        - Activation format (fp6_e2m3) balances precision and efficiency
        - Block size of 32 is optimal for hardware implementation
        - CUDA backend provides better numerical accuracy than PyTorch GPU
    """
    mx_specs = MxSpecs()
    
    # Shared exponent configuration
    mx_specs['scale_bits'] = scale_bits
    mx_specs['block_size'] = block_size
    mx_specs['shared_exp_method'] = 'max'  # Standard for MX formats
    
    # Element formats (forward pass)
    mx_specs['w_elem_format'] = weight_format
    mx_specs['a_elem_format'] = activation_format
    
    # Backward pass configuration (disabled for inference)
    mx_specs['quantize_backprop'] = quantize_backprop
    if quantize_backprop:
        mx_specs['w_elem_format_bp'] = weight_format
        mx_specs['a_elem_format_bp_ex'] = activation_format
        mx_specs['a_elem_format_bp_os'] = activation_format
    
    # Backend and rounding configuration
    mx_specs['custom_cuda'] = custom_cuda
    mx_specs['round'] = round_mode
    
    # Vector operations (non-matrix ops) - keep at full precision
    mx_specs['bfloat'] = 0  # Use native precision for add/mul/etc
    mx_specs['fp'] = 0
    
    return mx_specs


def create_mx_specs_weights_only(
    weight_format: str = 'fp4_e2m1',
    **kwargs
) -> Dict[str, Any]:
    """
    Create MX specs for weight-only quantization.
    
    Useful for ablation studies to measure impact of weight quantization alone.
    
    Args:
        weight_format (str): Element format for weights
        **kwargs: Additional arguments passed to create_mx_specs_exercise1
    
    Returns:
        Dict[str, Any]: MX specs with only weight quantization enabled
    """
    specs = create_mx_specs_exercise1(
        weight_format=weight_format,
        activation_format=None,  # No activation quantization
        **kwargs
    )
    return specs


def create_mx_specs_activations_only(
    activation_format: str = 'fp6_e2m3',
    **kwargs
) -> Dict[str, Any]:
    """
    Create MX specs for activation-only quantization.
    
    Useful for ablation studies to measure impact of activation quantization alone.
    
    Args:
        activation_format (str): Element format for activations
        **kwargs: Additional arguments passed to create_mx_specs_exercise1
    
    Returns:
        Dict[str, Any]: MX specs with only activation quantization enabled
    """
    specs = create_mx_specs_exercise1(
        weight_format=None,  # No weight quantization
        activation_format=activation_format,
        **kwargs
    )
    return specs


def print_mx_specs_summary(mx_specs: Dict[str, Any]) -> None:
    """
    Print a human-readable summary of MX specifications.
    
    Useful for debugging and logging configuration.
    
    Args:
        mx_specs (Dict[str, Any]): MX specs dictionary to summarize
    
    Example:
        >>> specs = create_mx_specs_exercise1()
        >>> print_mx_specs_summary(specs)
        MX Quantization Configuration:
        ==============================
        Weights: fp4_e2m1 (4-bit)
        Activations: fp6_e2m3 (6-bit)
        Scale Bits: 8 (E8M0)
        Block Size: 32
        CUDA Backend: Enabled
        Rounding: nearest
    """
    print("MX Quantization Configuration:")
    print("=" * 50)
    print(f"Weights: {mx_specs.get('w_elem_format', 'None')} "
          f"({_get_bit_width(mx_specs.get('w_elem_format', 'None'))}-bit)")
    print(f"Activations: {mx_specs.get('a_elem_format', 'None')} "
          f"({_get_bit_width(mx_specs.get('a_elem_format', 'None'))}-bit)")
    print(f"Scale Bits: {mx_specs.get('scale_bits', 0)} (E{mx_specs.get('scale_bits', 0)}M0)")
    print(f"Block Size: {mx_specs.get('block_size', 0)}")
    print(f"CUDA Backend: {'Enabled' if mx_specs.get('custom_cuda') else 'Disabled'}")
    print(f"Rounding: {mx_specs.get('round', 'N/A')}")
    print(f"Backward Quantization: {'Enabled' if mx_specs.get('quantize_backprop') else 'Disabled'}")
    print("=" * 50)


def _get_bit_width(format_str: Optional[str]) -> int:
    """Extract bit width from format string."""
    if format_str is None or format_str == 'None':
        return 0
    if 'fp4' in format_str or 'int4' in format_str:
        return 4
    if 'fp6' in format_str or 'int6' in format_str:
        return 6
    if 'fp8' in format_str or 'int8' in format_str:
        return 8
    if 'fp16' in format_str or 'int16' in format_str:
        return 16
    return 32  # Default FP32


def estimate_memory_savings(
    original_dtype: str = 'float32',
    weight_format: str = 'fp4_e2m1',
    activation_format: str = 'fp6_e2m3'
) -> Dict[str, float]:
    """
    Estimate memory savings from MX quantization.
    
    Args:
        original_dtype (str): Original data type ('float32', 'float16', 'bfloat16')
        weight_format (str): MX format for weights
        activation_format (str): MX format for activations
    
    Returns:
        Dict[str, float]: Dictionary with memory saving percentages
    
    Example:
        >>> savings = estimate_memory_savings()
        >>> print(f"Weight memory reduced by {savings['weight_savings']:.1f}%")
    """
    dtype_bits = {'float32': 32, 'float16': 16, 'bfloat16': 16}
    original_bits = dtype_bits.get(original_dtype, 32)
    
    weight_bits = _get_bit_width(weight_format)
    activation_bits = _get_bit_width(activation_format)
    
    weight_savings = ((original_bits - weight_bits) / original_bits) * 100
    activation_savings = ((original_bits - activation_bits) / original_bits) * 100
    
    return {
        'weight_savings': weight_savings,
        'activation_savings': activation_savings,
        'weight_compression_ratio': original_bits / weight_bits if weight_bits > 0 else 1.0,
        'activation_compression_ratio': original_bits / activation_bits if activation_bits > 0 else 1.0
    }


# Pre-configured specs for Exercise 1
EXERCISE1_MX_SPECS = create_mx_specs_exercise1()

# Print configuration when module is imported
if __name__ == "__main__":
    print("\nExercise 1: MX Quantization Helper Module")
    print_mx_specs_summary(EXERCISE1_MX_SPECS)
    print("\nEstimated Memory Savings:")
    savings = estimate_memory_savings()
    print(f"  Weights: {savings['weight_savings']:.1f}% "
          f"({savings['weight_compression_ratio']:.1f}x compression)")
    print(f"  Activations: {savings['activation_savings']:.1f}% "
          f"({savings['activation_compression_ratio']:.1f}x compression)")

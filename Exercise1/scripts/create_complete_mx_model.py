#!/usr/bin/env python3
"""
Script to create complete MX-integrated modeling_llama.py
==========================================================

This script downloads the original modeling_llama.py from transformers v4.57.6
and integrates our MX-quantized LlamaMLP and LlamaAttention classes.

Author: Pavan Chauhan
Date: January 29, 2026
"""

import urllib.request
import ssl
import re
import sys
import os

# URLs
TRANSFORMERS_VERSION = "v4.57.6"
ORIGINAL_URL = f"https://raw.githubusercontent.com/huggingface/transformers/{TRANSFORMERS_VERSION}/src/transformers/models/llama/modeling_llama.py"

# SSL context for downloads
ssl_context = ssl._create_unverified_context()

# Paths
TEMPLATE_FILE = "../modified_files/modeling_llama_mx_template.py"
OUTPUT_FILE = "../modified_files/modeling_llama.py"

def download_original():
    """Download original modeling_llama.py from transformers repo."""
    print(f"Downloading original modeling_llama.py from transformers {TRANSFORMERS_VERSION}...")
    try:
        with urllib.request.urlopen(ORIGINAL_URL, context=ssl_context) as response:
            content = response.read().decode('utf-8')
        print(f"✓ Downloaded {len(content)} bytes")
        return content
    except Exception as e:
        print(f"✗ Error downloading: {e}")
        sys.exit(1)

def extract_class(content, class_name):
    """Extract a complete class definition from Python code."""
    # Find class definition
    pattern = rf'class {class_name}\([^)]+\):\s*\n'
    match = re.search(pattern, content)
    
    if not match:
        return None
    
    start = match.start()
    lines = content[start:].split('\n')
    
    # Find where class ends (next class or end of file)
    class_lines = [lines[0]]  # First line is class definition
    indent_level = len(lines[1]) - len(lines[1].lstrip())
    
    for line in lines[1:]:
        # Check if we've reached another top-level class or function
        if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
            break
        # Check if we've reached a new class at the same level
        if line.strip().startswith('class ') and len(line) - len(line.lstrip()) == 0:
            break
        class_lines.append(line)
    
    return '\n'.join(class_lines)

def read_template():
    """Read our MX template file."""
    print(f"\nReading template file: {TEMPLATE_FILE}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, TEMPLATE_FILE)
    
    try:
        with open(template_path, 'r') as f:
            content = f.read()
        print(f"✓ Read {len(content)} bytes from template")
        return content
    except Exception as e:
        print(f"✗ Error reading template: {e}")
        sys.exit(1)

def create_integrated_file(original_content, template_content):
    """Create the complete integrated file."""
    print("\nIntegrating MX modifications...")
    
    # Define MX imports to add
    mx_imports_block = '''
# ============================================================================
# MX Quantization Setup - Exercise 1
# ============================================================================
import sys
import os

# Add microxcaling to path if needed
if '/content/microxcaling' not in sys.path:
    sys.path.insert(0, '/content/microxcaling')
if '/content/msr-intern-project/Exercise1' not in sys.path:
    sys.path.insert(0, '/content/msr-intern-project/Exercise1')

# MX library imports for quantization
try:
    from mx.specs import MxSpecs
    from mx import linear as mx_linear
    MX_AVAILABLE = True
except ImportError:
    import warnings
    warnings.warn("MX library not found. Running without quantization.")
    MX_AVAILABLE = False
    MxSpecs = dict
    mx_linear = None

# Exercise 1 MX configuration helper
try:
    from mx_config_helper import create_mx_specs_exercise1
except ImportError:
    # Fallback if helper not available
    def create_mx_specs_exercise1():
        if not MX_AVAILABLE:
            return {}
        return {
            'scale_bits': 8,
            'block_size': 32,
            'w_elem_format': 'fp4_e2m1',
            'a_elem_format': 'fp6_e2m3',
            'custom_cuda': True,
            'quantize_backprop': False,
            'round': 'nearest'
        }

# MX Quantization flag - can be controlled via environment variable
USE_MX_QUANTIZATION = os.environ.get('USE_MX_QUANTIZATION', '1') == '1' and MX_AVAILABLE

# Get MX specs for Exercise 1
EXERCISE1_MX_SPECS = create_mx_specs_exercise1() if USE_MX_QUANTIZATION else None

if USE_MX_QUANTIZATION:
    logger.info("MX Quantization ENABLED for Exercise 1")
    logger.info(f"  - Weight format: {EXERCISE1_MX_SPECS.get('w_elem_format')}")
    logger.info(f"  - Activation format: {EXERCISE1_MX_SPECS.get('a_elem_format')}")
    logger.info(f"  - Block size: {EXERCISE1_MX_SPECS.get('block_size')}")
else:
    logger.info("MX Quantization DISABLED - using standard FP32")
# ============================================================================
'''
    
    # Extract MX classes from template
    print("  - Extracting MX-quantized LlamaMLP...")
    mx_mlp = extract_class(template_content, 'LlamaMLP')
    if not mx_mlp:
        print("    ✗ Could not extract LlamaMLP from template")
        sys.exit(1)
    print(f"    ✓ Extracted LlamaMLP ({len(mx_mlp)} bytes)")
    
    print("  - Extracting MX-quantized LlamaAttention...")
    mx_attention = extract_class(template_content, 'LlamaAttention')
    if not mx_attention:
        print("    ✗ Could not extract LlamaAttention from template")
        sys.exit(1)
    print(f"    ✓ Extracted LlamaAttention ({len(mx_attention)} bytes)")
    
    # Extract helper function from template
    print("  - Extracting apply_mx_linear helper...")
    helper_start = template_content.find('def apply_mx_linear')
    if helper_start == -1:
        print("    ✗ Could not find apply_mx_linear")
        sys.exit(1)
    helper_end = template_content.find('\nclass ', helper_start)
    mx_helper = template_content[helper_start:helper_end].strip()
    print(f"    ✓ Extracted helper function ({len(mx_helper)} bytes)")
    
    # Find and replace in original
    original_lines = original_content.split('\n')
    result_lines = []
    modifications_made = []
    
    i = 0
    imports_added = False
    helper_added = False
    
    while i < len(original_lines):
        line = original_lines[i]
        
        # Add MX imports after logger definition
        if 'logger = logging.get_logger(__name__)' in line and not imports_added:
            result_lines.append(line)
            result_lines.append('')
            result_lines.extend(mx_imports_block.split('\n'))
            imports_added = True
            modifications_made.append("Added MX imports and setup")
            i += 1
            continue
        
        # Add helper function before first class definition
        if ('@use_kernel_forward_from_hub("RMSNorm")' in line or 
            'class LlamaRMSNorm' in line or 
            'class LlamaMLP' in line) and not helper_added:
            result_lines.append(mx_helper)
            result_lines.append('')
            result_lines.append('')
            helper_added = True
            modifications_made.append("Added apply_mx_linear helper function")
        
        # Skip the @use_kernel_forward decorator before apply_mx_linear if it exists
        if '@use_kernel_forward_from_hub' in line and i + 1 < len(original_lines):
            if 'def apply_mx_linear' in original_lines[i + 1]:
                # Skip this decorator and the function
                while i < len(original_lines) and not original_lines[i].startswith('class '):
                    i += 1
                continue
        
        # Replace LlamaMLP
        if 'class LlamaMLP(' in line:
            result_lines.append(mx_mlp)
            result_lines.append('')
            # Skip original class
            j = i + 1
            indent_level = 0
            while j < len(original_lines):
                if original_lines[j].strip().startswith('class ') and not original_lines[j].strip().startswith('class '):
                    break
                if original_lines[j].strip() and not original_lines[j][0].isspace():
                    if 'class ' in original_lines[j]:
                        break
                j += 1
            modifications_made.append(f"Replaced LlamaMLP (skipped {j-i} lines)")
            i = j
            continue
        
        # Replace LlamaAttention  
        if 'class LlamaAttention(' in line:
            result_lines.append(mx_attention)
            result_lines.append('')
            # Skip original class
            j = i + 1
            while j < len(original_lines):
                if original_lines[j].strip() and not original_lines[j][0].isspace():
                    if 'class ' in original_lines[j]:
                        break
                j += 1
            modifications_made.append(f"Replaced LlamaAttention (skipped {j-i} lines)")
            i = j
            continue
        
        result_lines.append(line)
        i += 1
    
    integrated_content = '\n'.join(result_lines)
    
    print("\nModifications applied:")
    for mod in modifications_made:
        print(f"  ✓ {mod}")
    
    return integrated_content

def save_output(content):
    """Save the integrated file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, OUTPUT_FILE)
    
    print(f"\nSaving to: {output_path}")
    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"✓ Saved {len(content)} bytes")
        print(f"✓ File: {output_path}")
        return output_path
    except Exception as e:
        print(f"✗ Error saving: {e}")
        sys.exit(1)

def main():
    print("=" * 70)
    print("MX Model Integration Script")
    print("=" * 70)
    
    # Download original
    original_content = download_original()
    
    # Read template
    template_content = read_template()
    
    # Create integrated file
    integrated_content = create_integrated_file(original_content, template_content)
    
    # Save output
    output_path = save_output(integrated_content)
    
    print("\n" + "=" * 70)
    print("✓ Integration Complete!")
    print("=" * 70)
    print(f"\nThe complete MX-integrated modeling_llama.py has been created.")
    print(f"Location: {output_path}")
    print("\nThis file can now be used to replace the original modeling_llama.py")
    print("in the transformers installation during evaluation.")
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()

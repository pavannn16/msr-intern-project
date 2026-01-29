#!/bin/bash
# Exercise 1 Setup Script for Google Colab
# ==========================================
# This script sets up the environment for Exercise 1: Linear Layer Quantization
# with MX data formats on Llama-3.2-1B model.
#
# Author: Pavan Chauhan
# Date: January 29, 2026

set -e  # Exit on error

echo "=========================================="
echo "Exercise 1: MX Linear Layer Quantization"
echo "=========================================="
echo ""

# Check if running in Colab
if [ ! -d "/content" ]; then
    echo "Warning: This script is optimized for Google Colab"
    echo "Current directory: $(pwd)"
fi

# Define paths
WORK_DIR="/content"
TRANSFORMERS_DIR="${WORK_DIR}/transformers"
MX_DIR="${WORK_DIR}/microxcaling"
PROJECT_DIR="${WORK_DIR}/msr-intern-project"
EXERCISE_DIR="${PROJECT_DIR}/Exercise1"

echo "[1/7] Checking base dependencies..."
if [ ! -d "$TRANSFORMERS_DIR" ]; then
    echo "  → Transformers not found. Running base setup..."
    if [ ! -f "${PROJECT_DIR}/scripts/setup_colab.sh" ]; then
        echo "  Error: Base setup script not found!"
        echo "  Please run the baseline setup first."
        exit 1
    fi
    bash "${PROJECT_DIR}/scripts/setup_colab.sh"
else
    echo "  ✓ Transformers found"
fi

if [ ! -d "$MX_DIR" ]; then
    echo "  Error: Microxcaling library not found!"
    echo "  Please run the baseline setup first."
    exit 1
else
    echo "  ✓ Microxcaling found"
fi

echo ""
echo "[2/7] Setting up Python path..."
export PYTHONPATH="${MX_DIR}:${EXERCISE_DIR}:${PYTHONPATH}"
echo "  ✓ Added ${MX_DIR} to PYTHONPATH"
echo "  ✓ Added ${EXERCISE_DIR} to PYTHONPATH"

echo ""
echo "[3/7] Verifying MX library installation..."
python3 << EOF
import sys
sys.path.insert(0, '${MX_DIR}')
try:
    from mx.specs import MxSpecs
    from mx import linear as mx_linear
    print("  ✓ MX library imports successful")
except ImportError as e:
    print(f"  ✗ MX library import failed: {e}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "  Error: MX library verification failed"
    exit 1
fi

echo ""
echo "[4/7] Creating Exercise 1 directory structure..."
mkdir -p "${EXERCISE_DIR}/results"
mkdir -p "${EXERCISE_DIR}/modified_files"
mkdir -p "${EXERCISE_DIR}/scripts"
echo "  ✓ Directory structure created"

echo ""
echo "[5/7] Checking for modified modeling_llama.py..."
MODIFIED_FILE="${EXERCISE_DIR}/modified_files/modeling_llama.py"
if [ -f "$MODIFIED_FILE" ]; then
    echo "  ✓ Modified modeling_llama.py found"
    
    # Create backup of original
    ORIGINAL_FILE="${TRANSFORMERS_DIR}/src/transformers/models/llama/modeling_llama.py"
    BACKUP_FILE="${TRANSFORMERS_DIR}/src/transformers/models/llama/modeling_llama.py.backup"
    
    if [ ! -f "$BACKUP_FILE" ]; then
        echo "  → Creating backup of original file..."
        cp "$ORIGINAL_FILE" "$BACKUP_FILE"
        echo "  ✓ Backup created: modeling_llama.py.backup"
    fi
    
    echo "  → Copying MX-integrated file to transformers..."
    cp "$MODIFIED_FILE" "$ORIGINAL_FILE"
    echo "  ✓ MX-integrated modeling_llama.py deployed"
else
    echo "  ⚠ Modified file not found at: $MODIFIED_FILE"
    echo "  You need to create the modified modeling_llama.py first"
    echo "  Or pull it from the GitHub repository"
fi

echo ""
echo "[6/7] Testing modified model import..."
python3 << EOF
import sys
sys.path.insert(0, '${MX_DIR}')
sys.path.insert(0, '${EXERCISE_DIR}')
try:
    from transformers.models.llama.modeling_llama import LlamaModel
    print("  ✓ Modified LlamaModel imports successfully")
except Exception as e:
    print(f"  ⚠ Import warning: {e}")
    print("  This may be normal if model hasn't been modified yet")
EOF

echo ""
echo "[7/7] Verifying CUDA availability..."
python3 << EOF
import torch
if torch.cuda.is_available():
    print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  ✓ CUDA version: {torch.version.cuda}")
else:
    print("  ✗ CUDA not available!")
    print("  Please ensure GPU runtime is enabled in Colab")
EOF

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Environment Configuration:"
echo "  - Transformers: ${TRANSFORMERS_DIR}"
echo "  - Microxcaling: ${MX_DIR}"
echo "  - Exercise 1: ${EXERCISE_DIR}"
echo ""
echo "Next Steps:"
echo "  1. Verify HF_TOKEN is set: echo \$HF_TOKEN"
echo "  2. Run evaluation with Exercise 1 notebook or:"
echo ""
echo "     lm_eval --model hf \\"
echo "       --model_args pretrained=meta-llama/Llama-3.2-1B \\"
echo "       --tasks lambada_openai \\"
echo "       --device cuda \\"
echo "       --batch_size 32"
echo ""
echo "  3. Compare results with baseline (62.10% accuracy)"
echo ""
echo "Troubleshooting:"
echo "  - If imports fail: Check PYTHONPATH includes ${MX_DIR}"
echo "  - If CUDA errors: Ensure GPU runtime is selected"
echo "  - If model errors: Verify modified file was copied correctly"
echo ""
echo "To restore original model:"
echo "  cp ${TRANSFORMERS_DIR}/src/transformers/models/llama/modeling_llama.py.backup \\"
echo "     ${TRANSFORMERS_DIR}/src/transformers/models/llama/modeling_llama.py"
echo ""

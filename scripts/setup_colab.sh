#!/bin/bash
# Microsoft Research Internship - Colab Environment Setup Script
# This script sets up the complete environment for the MX quantization exercise

set -e  # Exit on error

echo "======================================"
echo "MSR Internship Exercise - Setup Script"
echo "======================================"
echo ""

# Check if running in Colab
if [ ! -d "/content" ]; then
    echo "Warning: This script is designed for Google Colab environment"
    echo "Current directory: $(pwd)"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Clone repositories
echo "[1/5] Cloning transformers repository..."
if [ ! -d "/content/transformers" ]; then
    git clone https://github.com/huggingface/transformers.git /content/transformers
    cd /content/transformers
    git checkout v4.57.6
    echo "✓ Transformers cloned and checked out to v4.57.6"
else
    echo "✓ Transformers already exists"
fi

echo ""
echo "[2/5] Cloning microxcaling repository..."
if [ ! -d "/content/microxcaling" ]; then
    git clone https://github.com/microsoft/microxcaling.git /content/microxcaling
    echo "✓ Microxcaling cloned"
else
    echo "✓ Microxcaling already exists"
fi

echo ""
echo "[3/5] Installing transformers..."
cd /content/transformers
pip install -e . -q
echo "✓ Transformers installed"

echo ""
echo "[4/5] Installing lm-eval and dependencies..."
pip install lm_eval ninja -q
echo "✓ lm-eval and ninja installed"

echo ""
echo "[5/5] Setting up environment variables..."
export PYTHONPATH=/content/microxcaling:$PYTHONPATH
echo "✓ PYTHONPATH set to include microxcaling"

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "⚠️  IMPORTANT: Set your HF_TOKEN before running evaluations:"
echo "   export HF_TOKEN=<your_huggingface_token>"
echo ""
echo "📝 To test the setup, run:"
echo "   lm_eval --model hf \\"
echo "     --model_args pretrained=meta-llama/Llama-3.2-1B \\"
echo "     --tasks lambada_openai \\"
echo "     --device cuda \\"
echo "     --batch_size 32"
echo ""
echo "💡 Tip: Use --limit 0.1 for quick testing (10% of dataset)"
echo ""

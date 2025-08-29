#!/bin/bash
# Colab Setup Script for PCM-LLM
# This script optimizes the environment for Google Colab free tier

echo "🚀 Setting up PCM-LLM for Google Colab..."

# Install system dependencies
apt-get update -qq
apt-get install -qq git

# Clone or update the repository
if [ ! -d "pcm-llm" ]; then
    git clone https://github.com/yourusername/pcm-llm.git
    cd pcm-llm
else
    cd pcm-llm
    git pull
fi

# Install Python dependencies with Colab optimizations
pip install -q -r requirements.txt

# Set environment variables for Colab
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0

# Enable CUDA optimizations
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "⚠️  No NVIDIA GPU detected, using CPU"
fi

echo "🎯 PCM-LLM is ready for Colab!"
echo ""
echo "Quick start commands:"
echo "  cd pcm-llm"
echo "  python main.py"
echo ""
echo "For faster execution on limited resources:"
echo "  - Uses 4-bit quantization by default"
echo "  - Memory monitoring enabled"
echo "  - Aggressive memory cleanup"
echo "  - Optimized for small batch sizes"

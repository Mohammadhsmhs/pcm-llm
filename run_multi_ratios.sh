#!/bin/bash

# Quick multi-ratio benchmark runner
# Usage: ./run_multi_ratios.sh

echo "🚀 Starting multi-ratio benchmark..."

# Choose one of these 5-ratio configurations:

# Option 1: Comprehensive range (recommended)
ratios=(0.2 0.4 0.5 0.7 0.85)

# Option 2: Focus on aggressive compression
# ratios=(0.15 0.25 0.35 0.5 0.65)

# Option 3: Focus on conservative compression  
# ratios=(0.5 0.6 0.7 0.8 0.9)

# Option 4: Even distribution
# ratios=(0.2 0.35 0.5 0.65 0.8)

# Option 5: Research-focused (covers key thresholds)
# ratios=(0.25 0.4 0.6 0.75 0.9)

echo "📊 Testing ${#ratios[@]} compression ratios: ${ratios[*]}"
echo "🔬 This will provide enhanced statistical analysis with better trend detection"

# One-liner to test multiple compression ratios
for ratio in "${ratios[@]}"; do 
    echo ""
    echo "🎯 Testing compression ratio: $ratio"
    echo "================================"
    export PCM_COMPRESSION_TARGET_RATIO=$ratio
    pcm-llm all --sample 40
    if [ $? -ne 0 ]; then
        echo "❌ Failed at ratio $ratio, stopping..."
        exit 1
    fi
done

echo ""
echo "🎉 All ratios completed! Running comprehensive multi-ratio analysis..."
python analyzer_v2.py

echo "✅ Done! Check analysis_output/ for multi-ratio comparative results."

# Unlimited Mode Configuration

## Overview
The PCM-LLM benchmarking system includes an **Unlimited Mode** that removes all timeout and size restrictions for full-speed benchmarking. This is useful when you want to:

- Allow very long prompts to process completely
- Generate extended responses without token limits
- Run benchmarks without time constraints
- Test the absolute limits of your system

## ⚠️ Important Warning
**Unlimited Mode can cause:**
- Very long run times (hours or days)
- High memory and CPU usage
- Large output files
- Potential system instability

Use with caution and monitor your system resources.

## How to Enable

### Option 1: Quick Enable
Edit `config.py` and change:
```python
UNLIMITED_MODE = False  # Change to True
```
to:
```python
UNLIMITED_MODE = True   # Unlimited mode enabled
```

### Option 2: Runtime Override
For testing, you can override the setting in your code:
```python
import config
config.UNLIMITED_MODE = True  # Enable unlimited mode
```

## What Gets Disabled

### Timeouts
- **Evaluator timeouts**: No timeout limits on evaluation (normally 90s-300s based on prompt length)
- **LLM generation timeouts**: No timeout checks during response generation
- **Signal-based timeouts**: SIGALRM timeout signals are not set

### Size Limits
- **Max tokens (Llama.cpp)**: Increased from 512-1024 to 4096 tokens
- **Max new tokens (HuggingFace)**: Increased from 8192 to 16384 tokens
- **Response length limits**: No artificial truncation of responses

## Affected Components

### 1. Evaluation (`evaluation/evaluator.py`)
- Skips all timeout setup when unlimited mode is enabled
- Runs evaluations without time constraints

### 2. Llama.cpp LLM (`llms/llamacpp_llm.py`)
- Uses 1-hour timeout (but doesn't actively check it)
- Allows up to 4096 tokens in responses
- Skips timeout checks in streaming loop

### 3. HuggingFace LLM (`llms/huggingface_llm.py`)
- Increases max_new_tokens from 8192 to 16384
- No other timeout restrictions (HuggingFace handles its own limits)

### 4. Main Benchmark (`main.py`)
- Displays unlimited mode status at startup
- Shows warning messages about potential long run times

## Testing

Run the test script to verify unlimited mode:
```bash
python test_unlimited_mode.py
```

This will demonstrate the difference between unlimited and standard modes.

## Example Usage

```python
# Enable unlimited mode
import config
config.UNLIMITED_MODE = True

# Run your benchmark normally
from main import run_multi_task_benchmark
run_multi_task_benchmark(['reasoning', 'summarization'])
```

## Monitoring

When unlimited mode is active, monitor:
- System memory usage
- CPU utilization
- Disk space for results
- Process runtime

Consider using `top`, `htop`, or Activity Monitor to track resource usage.

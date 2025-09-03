"""
Data processing utilities for benchmark data handling.
"""

import csv
import os

from core.config import settings


def write_intermediate_csv(data_rows, csv_file_path, fieldnames):
    """Write data rows to intermediate CSV file."""
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    try:
        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                quoting=csv.QUOTE_ALL,
                escapechar="\\",
                doublequote=True,
            )
            writer.writeheader()
            writer.writerows(data_rows)
        print(f"💾 Intermediate file saved: {csv_file_path}")
    except Exception as e:
        print(f"⚠️  Failed to write intermediate CSV: {e}")


def extract_task_data(task_name: str, dataset):
    """Extract prompts and ground truths from dataset based on task type."""
    prompts = []
    ground_truths = []

    if task_name == "reasoning":
        for item in dataset:
            prompts.append(item["question"])
            ground_truths.append(item["answer"])
    elif task_name == "summarization":
        for item in dataset:
            prompts.append(item["article"])
            ground_truths.append(item["highlights"])
    elif task_name == "classification":
        for item in dataset:
            prompts.append(item["text"])
            ground_truths.append(item["label"])
    else:
        raise ValueError(f"Unsupported task: {task_name}")

    return prompts, ground_truths


def initialize_sample_result(
    sample_id: int, task_name: str, original_prompt: str, ground_truth: str
):
    """Initialize a sample result dictionary."""
    return {
        "sample_id": sample_id,
        "task": task_name,
        "llm_provider": settings.default_llm_provider,
        "llm_model": settings.get_llm_config(settings.default_llm_provider).model_name,
        "original_prompt": original_prompt,
        "ground_truth_answer": ground_truth,
        "compression_methods": [],
        "target_compression_ratio": 0.8,  # Will be updated
    }


def get_model_name(provider: str):
    """Get model name based on provider."""
    llm_config = settings.get_llm_config(provider)
    return llm_config.model_name

"""
Checkpoint management utilities for saving and loading benchmark progress.
"""

import json
import os

from core.config import settings


def save_checkpoint(checkpoint_data, checkpoint_file):
    """Save checkpoint data to file."""
    if not settings.evaluation.enable_checkpointing:
        return

    try:
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"💾 Checkpoint saved: {checkpoint_file}")
    except Exception as e:
        print(f"⚠️  Failed to save checkpoint: {e}")


def load_checkpoint(checkpoint_file):
    """Load checkpoint data from file."""
    if not settings.evaluation.enable_checkpointing or not os.path.exists(
        checkpoint_file
    ):
        return None

    try:
        with open(checkpoint_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Failed to load checkpoint: {e}")
        return None

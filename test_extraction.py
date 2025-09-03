#!/usr/bin/env python3
"""
Simple test script for the unified extraction function.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import just the extraction function
import re

def extract_structured_answer(text: str, task_type: str = "reasoning") -> str:
    """
    Extracts answers from structured responses using the #### format.
    This is now the unified extraction method since we enforce #### format across all tasks.
    """
    if not text:
        return ""

    text = text.strip()

    # Look for the #### format - this is now our primary and only extraction method
    structured_pattern = r"####\s*(.+?)(?:\n|$)"
    match = re.search(structured_pattern, text, re.DOTALL)
    if match:
        answer = match.group(1).strip()

        # Task-specific post-processing
        if task_type == "reasoning":
            # For reasoning, extract just the number if it's a math problem
            numbers = re.findall(r'[0-9.]+', answer)
            if numbers:
                return numbers[-1].strip('.')
            return answer
        elif task_type == "classification":
            # Normalize classification answers
            answer_lower = answer.lower()
            if answer_lower in ['positive', '1', 'pos']:
                return '1'
            elif answer_lower in ['negative', '0', 'neg']:
                return '0'
            else:
                return answer_lower
        else:
            # For summarization and other tasks, return as-is
            return answer

    # Fallback: if no #### format found, return the entire text
    return text.strip()

if __name__ == "__main__":
    print("🧪 Testing Unified Extraction Function")
    print("=" * 50)

    test_cases = [
        ("This movie is fantastic! #### positive", "classification"),
        ("The calculation gives 42. #### 42", "reasoning"),
        ("Article summary here. #### This is a concise summary of the main points discussed.", "summarization"),
        ("No structured format here", "reasoning"),
        ("Complex reasoning with #### 123.45", "reasoning"),
    ]

    for i, (text, task) in enumerate(test_cases, 1):
        result = extract_structured_answer(text, task)
        print(f"Test {i}:")
        print(f"  Task: {task}")
        print(f"  Input: {text}")
        print(f"  Extracted: '{result}'")
        print()

    print("✅ Extraction tests completed!")

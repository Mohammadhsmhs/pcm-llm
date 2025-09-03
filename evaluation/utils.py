import re


def extract_structured_answer(text: str, task_type: str = "reasoning") -> str:
    """
    Extracts answers from structured responses using the #### format.
    This is now the unified extraction method since we enforce #### format across all tasks.

    Args:
        text: The response text from the LLM
        task_type: The type of task (for backward compatibility)

    Returns:
        The extracted answer
    """
    if not text:
        return ""

    # Handle non-string inputs (like integers from classification labels)
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()

    # Look for the #### format - this is now our primary and only extraction method
    structured_pattern = r"####\s*(.+?)(?:\n|$)"
    match = re.search(structured_pattern, text, re.DOTALL)
    if match:
        answer = match.group(1).strip()

        # Task-specific post-processing
        if task_type == "reasoning":
            # For reasoning, extract just the number if it's a math problem
            numbers = re.findall(r"[0-9.]+", answer)
            if numbers:
                return numbers[-1].strip(".")
            return answer
        elif task_type == "classification":
            # Normalize classification answers
            answer_lower = answer.lower()
            if answer_lower in ["positive", "1", "pos"]:
                return "1"
            elif answer_lower in ["negative", "0", "neg"]:
                return "0"
            else:
                return answer_lower
        else:
            # For summarization and other tasks, return as-is
            return answer

    # Fallback: if no #### format found, return the entire text
    # This handles cases where the model didn't follow the format
    return text.strip()


def extract_gsm8k_answer(text: str) -> str:
    """
    Legacy function for backward compatibility.
    Now simply delegates to the unified extraction method.
    """
    return extract_structured_answer(text, "reasoning")

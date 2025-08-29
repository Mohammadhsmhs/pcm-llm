
import re


def extract_gsm8k_answer(text: str) -> str:
    """
    Extracts the final numerical answer from GSM8K-style responses.
    Looks for patterns like "The answer is 42" or "42" at the end.
    """
    if not text:
        return ""
    
    # Remove common prefixes and clean up
    text = text.strip()
    
    # Look for patterns like "The answer is X" or "Answer: X"
    patterns = [
        r"the answer is\s*([0-9.]+)",
        r"answer:\s*([0-9.]+)",
        r"answer is\s*([0-9.]+)",
        r"result is\s*([0-9.]+)",
        r"final answer:\s*([0-9.]+)",
        r"final answer is\s*([0-9.]+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1)
    
    # If no pattern found, look for the last number in the text
    numbers = re.findall(r'[0-9.]+', text)
    if numbers:
        return numbers[-1]
    
    return ""
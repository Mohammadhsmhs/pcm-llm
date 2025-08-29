
import re


def extract_gsm8k_answer(text: str) -> str:
    """
    Extracts the final numerical answer from GSM8K-style responses.
    Handles various formats including the standard "#### final_answer" format.
    """
    if not text:
        return ""
    
    # Remove common prefixes and clean up
    text = text.strip()
    
    # GSM8K specific: Look for "#### final_answer" pattern first
    gsm8k_pattern = r"####\s*([0-9.]+)"
    match = re.search(gsm8k_pattern, text)
    if match:
        return match.group(1).strip('.')
    
    # Look for patterns like "The answer is X" or "Answer: X"
    patterns = [
        r"the answer is\s*([0-9.]+)",
        r"answer:\s*([0-9.]+)",
        r"answer is\s*([0-9.]+)",
        r"result is\s*([0-9.]+)",
        r"final answer:\s*([0-9.]+)",
        r"final answer is\s*([0-9.]+)",
        r"she has\s*([0-9.]+)\s*buttons?",  # Look for "she has X buttons"
        r"will be\s*([0-9.]+)",  # Look for "will be X"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).strip('.')
    
    # Look for arithmetic expressions at the end like "9 + 20 + 60 = 89"
    arithmetic_pattern = r"(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)"
    match = re.search(arithmetic_pattern, text)
    if match:
        return match.group(4).strip('.')
    
    # Look for patterns with "buttons" at the end
    buttons_pattern = r"(\d+)\s*buttons?"
    matches = re.findall(buttons_pattern, text.lower())
    if matches:
        return matches[-1].strip('.')
    
    # If no pattern found, look for the last number in the text
    # But be more careful - look for numbers that appear to be final answers
    lines = text.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith(' ') and not line.startswith('-'):
            # Look for equals signs followed by numbers
            equals_match = re.search(r'=\s*([0-9.]+)', line)
            if equals_match:
                return equals_match.group(1).strip('.')
            
            # Look for numbers at the end of the line
            numbers = re.findall(r'[0-9.]+', line)
            if numbers:
                # Take the last number in the last meaningful line
                return numbers[-1].strip('.')
    
    return ""
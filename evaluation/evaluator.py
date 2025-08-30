
import time
import signal
import re
from evaluation.utils import extract_gsm8k_answer
from llms.base import BaseLLM
from config import UNLIMITED_MODE


class Evaluator:
    """
    Handles the end-to-end process of evaluating a prompt's performance.
    Supports multiple task types: reasoning, summarization, translation, classification.
    """

    def __init__(self, task: str, llm: BaseLLM):
        if not isinstance(llm, BaseLLM):
            raise TypeError("llm must be an instance of BaseLLM")
        self.task = task
        self.llm = llm

    def evaluate(self, prompt: str, ground_truth: str) -> dict:
        """
        Sends the prompt to the LLM and evaluates the response, returning key metrics.
        Includes timeout protection for long-running evaluations (unless unlimited mode is enabled).
        """
        def timeout_handler(signum, frame):
            raise TimeoutError("Evaluation timed out")
        
        # Skip timeout setup if unlimited mode is enabled
        if UNLIMITED_MODE:
            print("🔓 Unlimited mode: No timeout restrictions")
            try:
                start_time = time.time()
                response = self.llm.get_response(prompt)
                end_time = time.time()
                latency = end_time - start_time
                
                # Calculate task-specific metrics
                performance_score, extracted_answer = self._calculate_performance(response, ground_truth)
                
                return {
                    "score": performance_score,
                    "latency": round(latency, 2),
                    "llm_response": response,
                    "extracted_answer": extracted_answer,
                }
            except Exception as e:
                print(f"❌ Evaluation failed in unlimited mode: {e}")
                return {
                    "score": 0.0,
                    "latency": 0.0,
                    "llm_response": f"Error: Evaluation failed - {e}",
                    "extracted_answer": None,
                }
        
        # Standard timeout logic (when unlimited mode is disabled)
        # Set timeout based on prompt length and task type
        prompt_length = len(prompt.split())
        
        # More generous timeouts for realistic benchmarking
        if self.task == "summarization" and prompt_length > 1500:
            timeout_seconds = 300  # 5 minutes for very long summarization tasks
        elif prompt_length > 1500:
            timeout_seconds = 240  # 4 minutes for very long prompts
        elif prompt_length > 1000:
            timeout_seconds = 180  # 3 minutes for long prompts
        elif prompt_length > 500:
            timeout_seconds = 120  # 2 minutes for medium prompts
        else:
            timeout_seconds = 90   # 1.5 minutes for normal prompts
            
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        try:
            start_time = time.time()
            response = self.llm.get_response(prompt)
            end_time = time.time()
            latency = end_time - start_time
            
            # Calculate task-specific metrics
            performance_score, extracted_answer = self._calculate_performance(response, ground_truth)
            
            return {
                "score": performance_score,
                "latency": round(latency, 2),
                "llm_response": response,
                "extracted_answer": extracted_answer,
            }
        except TimeoutError:
            print(f"⚠️  Evaluation timed out after {timeout_seconds}s for prompt ({prompt_length} words)")
            return {
                "score": 0.0,
                "latency": timeout_seconds,
                "llm_response": f"Error: Evaluation timed out after {timeout_seconds} seconds",
                "extracted_answer": None,
            }
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def _calculate_performance(self, response: str, ground_truth: str):
        """Calculates a performance score based on the task."""
        if self.task == "reasoning":
            # For gsm8k, we need to extract the final number from both
            # the model's response and the ground truth label.
            response_answer = extract_gsm8k_answer(response)
            ground_truth_answer = extract_gsm8k_answer(ground_truth)

            if response_answer and ground_truth_answer:
                return 1.0 if response_answer == ground_truth_answer else 0.0, response_answer
            else:
                # Fallback to simple string matching if extraction fails
                return 1.0 if ground_truth in response else 0.0, response_answer

        elif self.task == "summarization":
            # Calculate ROUGE-like score for summarization
            return self._calculate_summarization_score(response, ground_truth), response

        elif self.task == "translation":
            # Calculate BLEU-like score for translation
            return self._calculate_translation_score(response, ground_truth), response

        elif self.task == "classification":
            # Calculate accuracy for classification
            return self._calculate_classification_score(response, ground_truth), response

        else:
            return 0.0, None

    def _calculate_summarization_score(self, response: str, ground_truth: str):
        """Calculate ROUGE-like score for summarization tasks."""
        # Simple ROUGE-1 implementation (unigram overlap)
        response_words = set(self._preprocess_text(response).split())
        ground_truth_words = set(self._preprocess_text(ground_truth).split())

        if not response_words or not ground_truth_words:
            return 0.0

        overlap = len(response_words.intersection(ground_truth_words))
        precision = overlap / len(response_words) if response_words else 0
        recall = overlap / len(ground_truth_words) if ground_truth_words else 0

        # F1 score
        if precision + recall == 0:
            return 0.0
        f1_score = 2 * (precision * recall) / (precision + recall)

        return f1_score

    def _calculate_translation_score(self, response: str, ground_truth: str):
        """Calculate BLEU-like score for translation tasks."""
        # Simple BLEU-1 implementation (unigram precision)
        response_words = self._preprocess_text(response).split()
        ground_truth_words = self._preprocess_text(ground_truth).split()

        if not response_words:
            return 0.0

        # Count matching words
        response_counts = {}
        ground_truth_counts = {}

        for word in response_words:
            response_counts[word] = response_counts.get(word, 0) + 1

        for word in ground_truth_words:
            ground_truth_counts[word] = ground_truth_counts.get(word, 0) + 1

        # Calculate clipped counts
        clipped_counts = 0
        for word, count in response_counts.items():
            clipped_counts += min(count, ground_truth_counts.get(word, 0))

        # BLEU precision
        if len(response_words) == 0:
            return 0.0

        precision = clipped_counts / len(response_words)

        # Length penalty (simplified)
        if len(response_words) < len(ground_truth_words):
            length_penalty = len(response_words) / len(ground_truth_words)
            precision *= length_penalty

        return precision

    def _calculate_classification_score(self, response: str, ground_truth: str):
        """Calculate accuracy for classification tasks."""
        # Extract predicted label from response
        response_lower = response.lower().strip()

        # Convert ground truth to int if it's a string
        if isinstance(ground_truth, str):
            ground_truth = int(ground_truth)

        # Look for explicit positive/negative answers first
        if 'positive' in response_lower and 'negative' not in response_lower:
            predicted = 1
        elif 'negative' in response_lower and 'positive' not in response_lower:
            predicted = 0
        else:
            # Look for positive/negative indicators
            positive_keywords = ['good', 'excellent', 'great', 'wonderful', 'amazing', 'fantastic', 'like', 'love', 'enjoy']
            negative_keywords = ['bad', 'terrible', 'awful', 'horrible', 'poor', 'disappointing', 'hate', 'dislike']

            has_positive = any(keyword in response_lower for keyword in positive_keywords)
            has_negative = any(keyword in response_lower for keyword in negative_keywords)

            if has_positive and not has_negative:
                predicted = 1  # positive
            elif has_negative and not has_positive:
                predicted = 0  # negative
            else:
                # Check for numbers or other patterns
                if '1' in response_lower and '0' not in response_lower:
                    predicted = 1
                elif '0' in response_lower and '1' not in response_lower:
                    predicted = 0
                else:
                    # Look for the first occurrence of positive/negative in the response
                    pos_idx = response_lower.find('positive')
                    neg_idx = response_lower.find('negative')
                    
                    if pos_idx != -1 and (neg_idx == -1 or pos_idx < neg_idx):
                        predicted = 1
                    elif neg_idx != -1 and (pos_idx == -1 or neg_idx < pos_idx):
                        predicted = 0
                    else:
                        predicted = None

        if predicted is None:
            print(f"⚠️  Could not determine sentiment from response: '{response[:100]}...'")
            return 0.0  # No clear prediction

        return 1.0 if predicted == ground_truth else 0.0

    def _preprocess_text(self, text: str):
        """Preprocess text for evaluation metrics."""
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

import re
import signal
import time

from core.config import settings
from evaluation.utils import extract_structured_answer
from llms.base import BaseLLM


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
        if settings.evaluation.unlimited_mode:
            print("🔓 Unlimited mode: No timeout restrictions")
            try:
                start_time = time.time()
                response = self.llm.get_response(prompt, self.task)
                end_time = time.time()
                latency = end_time - start_time

                # Calculate task-specific metrics
                performance_score, extracted_answer = self._calculate_performance(
                    response, ground_truth
                )

                # Calculate if answers match
                answers_match = self._calculate_answers_match(
                    extracted_answer, ground_truth
                )

                return {
                    "score": performance_score,
                    "latency": round(latency, 2),
                    "llm_response": response,
                    "extracted_answer": extracted_answer,
                    "answers_match": answers_match,
                }
            except Exception as e:
                print(f"❌ Evaluation failed in unlimited mode: {e}")
                return {
                    "score": 0.0,
                    "latency": 0.0,
                    "llm_response": f"Error: Evaluation failed - {e}",
                    "extracted_answer": None,
                    "answers_match": False,
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
            timeout_seconds = 90  # 1.5 minutes for normal prompts

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

        try:
            start_time = time.time()
            response = self.llm.get_response(prompt, self.task)
            end_time = time.time()
            latency = end_time - start_time

            # Calculate task-specific metrics
            performance_score, extracted_answer = self._calculate_performance(
                response, ground_truth
            )

            # Calculate if answers match
            answers_match = self._calculate_answers_match(
                extracted_answer, ground_truth
            )

            return {
                "score": performance_score,
                "latency": round(latency, 2),
                "llm_response": response,
                "extracted_answer": extracted_answer,
                "answers_match": answers_match,
            }
        except TimeoutError:
            print(
                f"⚠️  Evaluation timed out after {timeout_seconds}s for prompt ({prompt_length} words)"
            )
            return {
                "score": 0.0,
                "latency": timeout_seconds,
                "llm_response": f"Error: Evaluation timed out after {timeout_seconds} seconds",
                "extracted_answer": None,
                "answers_match": False,
            }
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return {
                "score": 0.0,
                "latency": 0.0,
                "llm_response": f"Error: Evaluation failed - {e}",
                "extracted_answer": None,
                "answers_match": False,
            }
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def _calculate_performance(self, response: str, ground_truth: str):
        """Calculates a performance score based on the task using unified structured extraction."""
        # Use the task type directly (no need for complex detection with structured format)
        task_type = self.task

        # Unified extraction and scoring for all task types
        response_answer = extract_structured_answer(response, task_type)

        if task_type == "classification":
            # For classification, ground_truth is already the expected answer
            ground_truth_answer = str(ground_truth).strip()
        elif task_type == "summarization":
            # For summarization, ground_truth is already the expected summary
            ground_truth_answer = str(ground_truth).strip()
        else:
            # For other tasks (reasoning, etc.), extract from ground_truth
            ground_truth_answer = extract_structured_answer(
                str(ground_truth), task_type
            )

        if task_type == "reasoning":
            # For reasoning, compare extracted answers
            if response_answer and ground_truth_answer:
                return (
                    1.0 if response_answer == ground_truth_answer else 0.0
                ), response_answer
            else:
                return 1.0 if ground_truth in response else 0.0, response_answer

        elif task_type == "classification":
            # For classification, compare normalized answers
            score = 1.0 if response_answer == ground_truth_answer else 0.0
            return score, response_answer

        elif task_type == "summarization":
            # For summarization, calculate ROUGE-like score
            score = self._calculate_summarization_score(
                response_answer, ground_truth_answer
            )
            if settings.evaluation.enable_qualitative_analysis:
                qualitative_feedback = self._analyze_summarization_quality(
                    response_answer, ground_truth_answer
                )
                print(f"📊 Summarization Quality Analysis: {qualitative_feedback}")
            return score, response_answer

        elif task_type == "translation":
            # For translation, calculate BLEU-like score
            return (
                self._calculate_translation_score(response_answer, ground_truth_answer),
                response_answer,
            )

        else:
            return 0.0, response_answer

    def _calculate_answers_match(
        self, extracted_answer: str, ground_truth: str
    ) -> bool:
        """Calculate whether the extracted answer matches the ground truth."""
        if not extracted_answer or not ground_truth:
            return False

        # For all tasks, compare the extracted answers directly
        task_type = self.task

        # Normalize both answers for comparison
        if task_type == "classification":
            # For classification, normalize to 0/1 and compare directly
            extracted_norm = extracted_answer.strip().lower()
            ground_truth_norm = str(ground_truth).strip().lower()

            # Convert common variations to standard format
            if extracted_norm in ["positive", "1", "pos", "true"]:
                extracted_norm = "1"
            elif extracted_norm in ["negative", "0", "neg", "false"]:
                extracted_norm = "0"

            if ground_truth_norm in ["positive", "1", "pos", "true"]:
                ground_truth_norm = "1"
            elif ground_truth_norm in ["negative", "0", "neg", "false"]:
                ground_truth_norm = "0"

            return extracted_norm == ground_truth_norm

        elif task_type == "reasoning":
            # For reasoning, extract answer from ground truth first, then compare
            ground_truth_answer = extract_structured_answer(
                str(ground_truth), task_type
            )
            return extracted_answer.strip() == ground_truth_answer.strip()

        elif task_type == "summarization":
            # For summarization, check if extracted answer contains key information
            # This is a simplified check - in practice you'd want more sophisticated matching
            extracted_lower = extracted_answer.lower().strip()
            ground_truth_lower = ground_truth.lower().strip()

            # Check for substantial overlap (at least 50% of ground truth words)
            extracted_words = set(extracted_lower.split())
            ground_truth_words = set(ground_truth_lower.split())

            if not ground_truth_words:
                return False

            overlap = len(extracted_words.intersection(ground_truth_words))
            return overlap / len(ground_truth_words) >= 0.5

        else:
            # Default: extract answer from ground truth first, then compare
            ground_truth_answer = extract_structured_answer(ground_truth, task_type)
            return extracted_answer.strip() == ground_truth_answer.strip()

    def _detect_task_type(self, ground_truth: str):
        """Detect task type from ground truth format."""
        # Reasoning: Contains mathematical expressions or numbers
        if (
            any(char.isdigit() for char in ground_truth)
            or "calculate" in ground_truth.lower()
        ):
            return "reasoning"

        # Classification: Usually just "0" or "1" or short sentiment words
        if ground_truth.strip() in ["0", "1", "positive", "negative"]:
            return "classification"

        # Summarization: Longer text, typically contains multiple sentences
        if len(ground_truth.split()) > 10:
            return "summarization"

        # Default to reasoning if unclear
        return "reasoning"

    def _calculate_summarization_score(self, response: str, ground_truth: str):
        """Calculate ROUGE-like score for summarization tasks with style-aware scoring."""
        # Preprocess both texts
        response_words = set(self._preprocess_text(response).split())
        ground_truth_words = set(self._preprocess_text(ground_truth).split())

        if not response_words or not ground_truth_words:
            return 0.0

        # Calculate basic ROUGE-1 metrics
        overlap = len(response_words.intersection(ground_truth_words))
        precision = overlap / len(response_words) if response_words else 0
        recall = overlap / len(ground_truth_words) if ground_truth_words else 0

        # F1 score
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        # Style adjustment: Boost score for responses that are appropriately concise
        # Ground truth summaries are typically 1-2 sentences, so we reward similar brevity
        adjusted_score = f1_score

        if settings.evaluation.enable_style_aware_scoring:
            response_sentences = len([s for s in response.split(".") if s.strip()])
            ground_truth_sentences = len(
                [s for s in ground_truth.split(".") if s.strip()]
            )

            # Length similarity bonus (0.1 max bonus for similar sentence count)
            length_bonus = 0.0
            if abs(response_sentences - ground_truth_sentences) <= 1:
                length_bonus = 0.1
            elif abs(response_sentences - ground_truth_sentences) <= 2:
                length_bonus = 0.05

            # Apply length bonus to final score
            adjusted_score = min(1.0, f1_score + length_bonus)

            # Debug logging for style analysis
            print(f"🔍 Summarization Style Analysis:")
            print(
                f"   Response sentences: {response_sentences}, Ground truth sentences: {ground_truth_sentences}"
            )
            print(
                f"   ROUGE F1: {f1_score:.3f}, Length bonus: {length_bonus:.3f}, Final score: {adjusted_score:.3f}"
            )

        return adjusted_score

    def _analyze_summarization_quality(self, response: str, ground_truth: str):
        """Provide qualitative analysis of summarization quality."""
        analysis = []

        # Length analysis
        response_words = len(response.split())
        ground_truth_words = len(ground_truth.split())
        response_sentences = len([s for s in response.split(".") if s.strip()])
        # ground_truth_sentences = len([s for s in ground_truth.split(".") if s.strip()])

        # Style assessment
        if response_sentences <= 2 and response_words <= ground_truth_words * 1.5:
            analysis.append("✅ Appropriate brevity (similar to ground truth)")
        elif response_sentences <= 3 and response_words <= ground_truth_words * 2:
            analysis.append("⚠️  Moderately verbose (acceptable for research)")
        else:
            analysis.append("❌ Too verbose (may affect ROUGE scoring)")

        # Content assessment
        response_lower = response.lower()
        ground_truth_lower = ground_truth.lower()

        # Check for key information preservation
        key_terms = [word for word in ground_truth_lower.split() if len(word) > 4]
        preserved_terms = sum(1 for term in key_terms if term in response_lower)
        preservation_rate = preserved_terms / len(key_terms) if key_terms else 0

        if preservation_rate >= 0.7:
            analysis.append("✅ Good content preservation")
        elif preservation_rate >= 0.5:
            analysis.append("⚠️  Moderate content preservation")
        else:
            analysis.append("❌ Poor content preservation")

        # Factual consistency check
        if any(char.isdigit() for char in response) and any(
            char.isdigit() for char in ground_truth
        ):
            analysis.append("✅ Contains numerical facts")
        else:
            analysis.append("ℹ️  No numerical facts to verify")

        return " | ".join(analysis)

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

    def _preprocess_text(self, text: str):
        """Preprocess text for evaluation metrics."""
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

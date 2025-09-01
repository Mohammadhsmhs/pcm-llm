
import time
import signal
import re
from evaluation.utils import extract_gsm8k_answer
from llms.base import BaseLLM
from config import UNLIMITED_MODE, ENABLE_QUALITATIVE_ANALYSIS, ENABLE_STYLE_AWARE_SCORING


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
        # Handle multi-task evaluation by detecting task type from content
        if self.task == "multi":
            task_type = self._detect_task_type(ground_truth)
        else:
            task_type = self.task
            
        if task_type == "reasoning":
            # For gsm8k, we need to extract the final number from both
            # the model's response and the ground truth label.
            response_answer = extract_gsm8k_answer(response)
            ground_truth_answer = extract_gsm8k_answer(ground_truth)

            if response_answer and ground_truth_answer:
                return 1.0 if response_answer == ground_truth_answer else 0.0, response_answer
            else:
                # Fallback to simple string matching if extraction fails
                return 1.0 if ground_truth in response else 0.0, response_answer

        elif task_type == "summarization":
            # Calculate ROUGE-like score for summarization
            score = self._calculate_summarization_score(response, ground_truth)
            # Also provide qualitative analysis for summarization if enabled
            if ENABLE_QUALITATIVE_ANALYSIS:
                qualitative_feedback = self._analyze_summarization_quality(response, ground_truth)
                print(f"📊 Summarization Quality Analysis: {qualitative_feedback}")
            return score, response

        elif task_type == "translation":
            # Calculate BLEU-like score for translation
            return self._calculate_translation_score(response, ground_truth), response

        elif task_type == "classification":
            # Calculate accuracy for classification
            score, extracted_answer = self._calculate_classification_score(response, ground_truth)
            return score, extracted_answer

        else:
            return 0.0, None

    def _detect_task_type(self, ground_truth: str):
        """Detect task type from ground truth format."""
        # Reasoning: Contains mathematical expressions or numbers
        if any(char.isdigit() for char in ground_truth) or 'calculate' in ground_truth.lower():
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
        
        if ENABLE_STYLE_AWARE_SCORING:
            response_sentences = len([s for s in response.split('.') if s.strip()])
            ground_truth_sentences = len([s for s in ground_truth.split('.') if s.strip()])
            
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
            print(f"   Response sentences: {response_sentences}, Ground truth sentences: {ground_truth_sentences}")
            print(f"   ROUGE F1: {f1_score:.3f}, Length bonus: {length_bonus:.3f}, Final score: {adjusted_score:.3f}")
        
        return adjusted_score

    def _analyze_summarization_quality(self, response: str, ground_truth: str):
        """Provide qualitative analysis of summarization quality."""
        analysis = []
        
        # Length analysis
        response_words = len(response.split())
        ground_truth_words = len(ground_truth.split())
        response_sentences = len([s for s in response.split('.') if s.strip()])
        ground_truth_sentences = len([s for s in ground_truth.split('.') if s.strip()])
        
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
        if any(char.isdigit() for char in response) and any(char.isdigit() for char in ground_truth):
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

    def _calculate_classification_score(self, response: str, ground_truth: str):
        """Calculate accuracy for classification tasks."""
        # Extract predicted label from response
        response_lower = response.lower().strip()

        # Debug logging
        print(f"🔍 Classification Debug - Response: '{response}'")
        print(f"🔍 Classification Debug - Response lower: '{response_lower}'")
        print(f"🔍 Classification Debug - Ground truth: '{ground_truth}'")

        # Convert ground truth to int if it's a string
        if isinstance(ground_truth, str):
            ground_truth = int(ground_truth)

        # Look for explicit positive/negative answers first
        if 'positive' in response_lower and 'negative' not in response_lower:
            predicted = 1
            extracted_answer = "1"
            print(f"✅ Found 'positive' -> predicted: {predicted}, extracted: {extracted_answer}")
        elif 'negative' in response_lower and 'positive' not in response_lower:
            predicted = 0
            extracted_answer = "0"
            print(f"✅ Found 'negative' -> predicted: {predicted}, extracted: {extracted_answer}")
        else:
            print("⚠️  No direct positive/negative found, checking keywords...")
            # Look for positive/negative indicators
            positive_keywords = ['good', 'excellent', 'great', 'wonderful', 'amazing', 'fantastic', 'like', 'love', 'enjoy', 'best', 'favorite', 'recommend']
            negative_keywords = ['bad', 'terrible', 'awful', 'horrible', 'poor', 'disappointing', 'hate', 'dislike', 'worst', 'boring', 'waste']

            has_positive = any(keyword in response_lower for keyword in positive_keywords)
            has_negative = any(keyword in response_lower for keyword in negative_keywords)

            print(f"🔍 Has positive keywords: {has_positive}, Has negative keywords: {has_negative}")

            if has_positive and not has_negative:
                predicted = 1  # positive
                extracted_answer = "1"
                print(f"✅ Found positive keywords -> predicted: {predicted}, extracted: {extracted_answer}")
            elif has_negative and not has_positive:
                predicted = 0  # negative
                extracted_answer = "0"
                print(f"✅ Found negative keywords -> predicted: {predicted}, extracted: {extracted_answer}")
            else:
                print("⚠️  No clear keywords found, checking numbers...")
                # Check for numbers or other patterns
                if '1' in response_lower and '0' not in response_lower:
                    predicted = 1
                    extracted_answer = "1"
                    print(f"✅ Found '1' -> predicted: {predicted}, extracted: {extracted_answer}")
                elif '0' in response_lower and '1' not in response_lower:
                    predicted = 0
                    extracted_answer = "0"
                    print(f"✅ Found '0' -> predicted: {predicted}, extracted: {extracted_answer}")
                else:
                    print("⚠️  No numbers found, checking positions...")
                    # Look for the last occurrence of positive/negative in the response
                    pos_idx = response_lower.rfind('positive')
                    neg_idx = response_lower.rfind('negative')
                    
                    print(f"🔍 Position of 'positive': {pos_idx}, Position of 'negative': {neg_idx}")
                    
                    if pos_idx != -1 and (neg_idx == -1 or pos_idx > neg_idx):
                        predicted = 1
                        extracted_answer = "1"
                        print(f"✅ 'positive' found last -> predicted: {predicted}, extracted: {extracted_answer}")
                    elif neg_idx != -1 and (pos_idx == -1 or neg_idx > pos_idx):
                        predicted = 0
                        extracted_answer = "0"
                        print(f"✅ 'negative' found last -> predicted: {predicted}, extracted: {extracted_answer}")
                    else:
                        print("⚠️  No positive/negative found at all...")
                        # Check for neutral or mixed
                        neutral_keywords = ['neutral', 'mixed', 'average', 'okay', 'mediocre']
                        if any(keyword in response_lower for keyword in neutral_keywords):
                            # For neutral, predict 0 (negative) as default
                            predicted = 0
                            extracted_answer = "0"
                            print(f"✅ Found neutral keywords -> predicted: {predicted}, extracted: {extracted_answer}")
                        else:
                            predicted = None
                            extracted_answer = None
                            print("❌ Could not determine sentiment")

        if predicted is None:
            print(f"⚠️  Could not determine sentiment from response: '{response[:100]}...'")
            return 0.0, None  # No clear prediction

        final_score = 1.0 if predicted == ground_truth else 0.0
        print(f"🎯 Final result - Predicted: {predicted}, Ground truth: {ground_truth}, Score: {final_score}, Extracted: {extracted_answer}")
        return final_score, extracted_answer

    def _preprocess_text(self, text: str):
        """Preprocess text for evaluation metrics."""
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

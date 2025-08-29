
import time
from evaluation.utils import extract_gsm8k_answer
from llms.base import BaseLLM


class Evaluator:
    """
    Handles the end-to-end process of evaluating a prompt's performance.
    It orchestrates sending the prompt to the LLM and then scoring the response.
    """

    def __init__(self, task: str, llm: BaseLLM):
        if not isinstance(llm, BaseLLM):
            raise TypeError("llm must be an instance of BaseLLM")
        self.task = task
        self.llm = llm

    def evaluate(self, prompt: str, ground_truth: str) -> dict:
        """
        Sends the prompt to the LLM and evaluates the response, returning key metrics.
        """
        start_time = time.time()
        response = self.llm.get_response(prompt)
        end_time = time.time()

        latency = end_time - start_time

        # Calculate task-specific metrics
        performance_score = self._calculate_performance(response, ground_truth)

        return {
            "score": performance_score,
            "latency": round(latency, 2),
            "llm_response": response,
        }

    def _calculate_performance(self, response: str, ground_truth: str) -> float:
        """Calculates a performance score based on the task."""
        if self.task == "reasoning":
            # For gsm8k, we need to extract the final number from both
            # the model's response and the ground truth label.
            response_answer = extract_gsm8k_answer(response)
            ground_truth_answer = extract_gsm8k_answer(ground_truth)
            
            if response_answer and ground_truth_answer:
                return 1.0 if response_answer == ground_truth_answer else 0.0
            else:
                # Fallback to simple string matching if extraction fails
                return 1.0 if ground_truth in response else 0.0
        # elif self.task == "summarization":
        #     # Here you would implement ROUGE score calculation
        #     return calculate_rouge(response, ground_truth)
        else:
            return 0.0

class DataEnhancer:
    """
    Utility class for enhancing benchmark result data with calculated metrics.
    All methods are static, making it a stateless utility.
    """

    @staticmethod
    def enhance_result_data(result_data):
        """
        Calculate additional metrics from the basic result data.
        This method is static and can be called without creating an instance.
        """
        enhanced_data = result_data.copy()

        # Safely get values
        baseline_score = enhanced_data.get("baseline_score", 0.0)
        compressed_score = enhanced_data.get("compressed_score", 0.0)
        baseline_latency = enhanced_data.get("baseline_latency", 0.0)
        compressed_latency = enhanced_data.get("compressed_latency", 0.0)
        original_prompt = enhanced_data.get("original_prompt", "")
        compressed_prompt = enhanced_data.get("compressed_prompt", "")

        # --- Score & Quality Metrics ---
        # Score Preservation Ratio
        if baseline_score > 0:
            enhanced_data["score_preservation"] = (
                compressed_score / baseline_score
            ) * 100
        else:
            enhanced_data["score_preservation"] = (
                0.0 if compressed_score == 0 else 100.0
            )

        # Quality Degradation
        enhanced_data["quality_degradation"] = baseline_score - compressed_score
        enhanced_data["quality_degradation_percent"] = (
            enhanced_data["score_preservation"] - 100
        )

        # --- Latency Metrics ---
        # Latency Overhead
        latency_overhead = compressed_latency - baseline_latency
        enhanced_data["latency_overhead_seconds"] = latency_overhead
        if baseline_latency > 0:
            enhanced_data["latency_overhead_percent"] = (
                latency_overhead / baseline_latency
            ) * 100
        else:
            enhanced_data["latency_overhead_percent"] = 0.0

        # --- Compression & Token Metrics ---
        # Actual Compression Ratio & Tokens Saved
        original_tokens = len(original_prompt.split())
        compressed_tokens = len(compressed_prompt.split())

        if original_tokens > 0:
            actual_ratio = compressed_tokens / original_tokens
        else:
            actual_ratio = 0.0

        enhanced_data["actual_compression_ratio"] = actual_ratio * 100
        enhanced_data["tokens_saved"] = original_tokens - compressed_tokens

        # Compression Efficiency
        target_ratio = enhanced_data.get("target_compression_ratio", 0.8)
        if target_ratio > 0:
            # How well the compressor met the target ratio
            efficiency = (
                (1 - actual_ratio) / (1 - target_ratio) if target_ratio < 1 else 1.0
            )
            enhanced_data["compression_efficiency"] = (
                min(efficiency, 1.0) * 100
            )  # Cap at 100%
        else:
            enhanced_data["compression_efficiency"] = 0.0

        return enhanced_data

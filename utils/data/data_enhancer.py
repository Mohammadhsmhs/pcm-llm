from transformers import AutoTokenizer

class DataEnhancer:
    """
    Utility class for enhancing benchmark result data with calculated metrics.
    All methods are static, making it a stateless utility.
    """
    
    # Class-level tokenizer cache to avoid reloading
    _tokenizer_cache = {}

    @staticmethod
    def _get_tokenizer(model_name="bert-base-uncased"):
        """Get tokenizer with caching to avoid repeated loading."""
        if model_name not in DataEnhancer._tokenizer_cache:
            DataEnhancer._tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
        return DataEnhancer._tokenizer_cache[model_name]

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
        # Calculate actual ratios with proper tokenization and clear terminology
        tokenizer = DataEnhancer._get_tokenizer("bert-base-uncased")  # Match naive_truncation tokenizer
        
        try:
            # Use proper tokenization (same as compression methods)
            original_tokens = len(tokenizer.encode(original_prompt, add_special_tokens=False))
            compressed_tokens = len(tokenizer.encode(compressed_prompt, add_special_tokens=False))
        except Exception:
            # Fallback to space splitting if tokenization fails
            original_tokens = len(original_prompt.split())
            compressed_tokens = len(compressed_prompt.split())

        if original_tokens > 0:
            # FIXED TERMINOLOGY AND RATIOS:
            # preservation_ratio = what fraction of original tokens remain (0.0 to 1.0+)
            preservation_ratio = compressed_tokens / original_tokens
            # compression_ratio = what fraction was removed (0.0 to 1.0)
            compression_ratio = max(0.0, 1.0 - preservation_ratio)
        else:
            preservation_ratio = 0.0
            compression_ratio = 0.0

        # Store with CORRECT terminology and as RATIOS (not percentages)
        enhanced_data["actual_preservation_ratio"] = preservation_ratio  # Clear: what fraction preserved
        enhanced_data["actual_compression_ratio"] = compression_ratio    # Clear: what fraction compressed  
        enhanced_data["tokens_saved"] = original_tokens - compressed_tokens

        # Compression Efficiency
        # FIXED: target_ratio is actually a PRESERVATION ratio (0.8 = keep 80% of tokens)
        target_preservation_ratio = enhanced_data.get("target_preservation_ratio") or enhanced_data.get("target_compression_ratio")  # Try new key first, then fallback
        
        if target_preservation_ratio is None:
            # FIXED: Don't use hardcoded default - this should come from the actual benchmark run
            print(f"⚠️  Warning: target_preservation_ratio missing for sample, efficiency calculation skipped")
            enhanced_data["compression_efficiency"] = 0.0
            enhanced_data["target_preservation_ratio"] = None
        elif target_preservation_ratio > 0:
            # How well the compressor met the target preservation ratio
            # If target was to preserve 0.8 and we preserved 0.7, efficiency = 0.7/0.8 = 87.5%
            efficiency = (
                preservation_ratio / target_preservation_ratio if target_preservation_ratio > 0 else 1.0
            )
            enhanced_data["compression_efficiency"] = (
                min(efficiency, 1.0) * 100
            )  # Cap at 100%
            # FIXED: Store the correctly named target ratio
            enhanced_data["target_preservation_ratio"] = target_preservation_ratio
        else:
            enhanced_data["compression_efficiency"] = 0.0
            enhanced_data["target_preservation_ratio"] = target_preservation_ratio

        return enhanced_data

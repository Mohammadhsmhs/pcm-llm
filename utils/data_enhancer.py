class DataEnhancer:
    """
    Responsible for enhancing raw data with calculated metrics.
    Single Responsibility: Data enhancement and metric calculation.
    """

    def enhance_result_data(self, result_data):
        """Calculate additional metrics from basic result data."""
        enhanced = result_data.copy()

        # Calculate prompt lengths and compression metrics
        original_length = len(result_data.get('original_prompt', ''))
        compressed_length = len(result_data.get('compressed_prompt', ''))

        enhanced['original_prompt_length'] = original_length
        enhanced['compressed_prompt_length'] = compressed_length

        if original_length > 0:
            actual_ratio = compressed_length / original_length
            enhanced['actual_compression_ratio'] = round(actual_ratio, 4)
            enhanced['tokens_saved'] = original_length - compressed_length

            # Compression efficiency (how close we got to target)
            target_ratio = result_data.get('target_compression_ratio', 0.8)
            efficiency = 1 - abs(actual_ratio - target_ratio) / target_ratio
            enhanced['compression_efficiency'] = round(max(0, efficiency), 4)
        else:
            enhanced['actual_compression_ratio'] = 1.0
            enhanced['tokens_saved'] = 0
            enhanced['compression_efficiency'] = 0.0

        # Calculate score preservation
        baseline_score = result_data.get('baseline_score', 0)
        compressed_score = result_data.get('compressed_score', 0)
        if baseline_score > 0:
            enhanced['score_preservation'] = round(compressed_score / baseline_score, 4)
        else:
            enhanced['score_preservation'] = 0.0

        # Calculate latency overhead
        baseline_latency = result_data.get('baseline_latency', 0)
        compressed_latency = result_data.get('compressed_latency', 0)
        if baseline_latency > 0:
            overhead = (compressed_latency - baseline_latency) / baseline_latency
            enhanced['latency_overhead'] = round(overhead, 4)
        else:
            enhanced['latency_overhead'] = 0.0

        # Calculate quality degradation
        if baseline_score > 0:
            degradation = (baseline_score - compressed_score) / baseline_score
            enhanced['quality_degradation'] = round(max(0, degradation), 4)
        else:
            enhanced['quality_degradation'] = 0.0

        return enhanced

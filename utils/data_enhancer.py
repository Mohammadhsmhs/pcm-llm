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

        # Calculate latency overhead (absolute seconds and relative percent)
        baseline_latency = result_data.get('baseline_latency', 0)
        compressed_latency = result_data.get('compressed_latency', 0)
        if baseline_latency and isinstance(baseline_latency, (int, float)):
            overhead_seconds = (compressed_latency or 0) - baseline_latency
            enhanced['latency_overhead_seconds'] = round(overhead_seconds, 4)
            # Relative overhead (kept as existing field for backward compatibility)
            enhanced['latency_overhead'] = round(overhead_seconds / baseline_latency, 4) if baseline_latency else 0.0
            enhanced['latency_overhead_percent'] = round(100.0 * enhanced['latency_overhead'], 2)
        else:
            enhanced['latency_overhead_seconds'] = 0.0
            enhanced['latency_overhead'] = 0.0
            enhanced['latency_overhead_percent'] = 0.0

        # Calculate quality degradation
        # absolute_diff = baseline - compressed (negative means improvement)
        baseline_score = result_data.get('baseline_score', 0) or 0.0
        compressed_score = result_data.get('compressed_score', 0) or 0.0
        abs_delta = baseline_score - compressed_score
        enhanced['quality_degradation'] = round(abs_delta, 4)
        # relative percent (signed). Only meaningful if baseline_score != 0
        enhanced['quality_degradation_percent'] = round(100.0 * (abs_delta / baseline_score), 2) if baseline_score else 0.0

        return enhanced

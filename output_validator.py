"""
Output validation for LLM sentiment analysis results.

Validates that LLM outputs are complete and within expected ranges.
"""

import pandas as pd
from typing import Dict, Any, List, Tuple


class OutputValidator:
    """
    Validates LLM sentiment analysis outputs for completeness and correctness.
    """

    def __init__(self):
        """Initialize validator with expected field specifications."""
        self.required_fields = [
            'hawkish_dovish_score',
            'topics',
            'uncertainty',
            'forward_guidance_strength',
            'key_sentences',
            'market_impact',
            'summary'
        ]

        self.required_topic_fields = [
            'inflation',
            'growth',
            'financial_stability',
            'labor_market',
            'international'
        ]

        self.required_market_fields = [
            'stocks',
            'bonds',
            'currency',
            'reasoning'
        ]

        self.score_ranges = {
            'hawkish_dovish_score': (-100, 100),
            'uncertainty': (0, 100),
            'forward_guidance_strength': (0, 100),
            'topic_scores': (0, 100)
        }

        self.valid_market_values = ['rise', 'fall', 'neutral']

    def validate_single_output(self, output: Dict[str, Any],
                                speech_id: str = "unknown") -> Tuple[bool, List[str]]:
        """
        Validate a single LLM output.

        Args:
            output: Parsed JSON output from LLM
            speech_id: ID for error reporting

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check required fields
        for field in self.required_fields:
            if field not in output:
                errors.append(f"Missing required field: {field}")

        if errors:
            return False, errors

        # Validate hawkish/dovish score
        hd_score = output.get('hawkish_dovish_score')
        if not isinstance(hd_score, (int, float)):
            errors.append(f"hawkish_dovish_score must be numeric, got: {type(hd_score)}")
        elif hd_score < -100 or hd_score > 100:
            errors.append(f"hawkish_dovish_score out of range [-100, 100]: {hd_score}")

        # Validate topics
        topics = output.get('topics', {})
        if not isinstance(topics, dict):
            errors.append(f"topics must be a dictionary, got: {type(topics)}")
        else:
            for topic_field in self.required_topic_fields:
                if topic_field not in topics:
                    errors.append(f"Missing topic field: {topic_field}")
                else:
                    score = topics[topic_field]
                    if not isinstance(score, (int, float)):
                        errors.append(f"Topic '{topic_field}' must be numeric")
                    elif score < 0 or score > 100:
                        errors.append(f"Topic '{topic_field}' out of range [0, 100]: {score}")

        # Validate uncertainty
        uncertainty = output.get('uncertainty')
        if not isinstance(uncertainty, (int, float)):
            errors.append(f"uncertainty must be numeric")
        elif uncertainty < 0 or uncertainty > 100:
            errors.append(f"uncertainty out of range [0, 100]: {uncertainty}")

        # Validate forward guidance
        fg_strength = output.get('forward_guidance_strength')
        if not isinstance(fg_strength, (int, float)):
            errors.append(f"forward_guidance_strength must be numeric")
        elif fg_strength < 0 or fg_strength > 100:
            errors.append(f"forward_guidance_strength out of range [0, 100]: {fg_strength}")

        # Validate market impact
        market_impact = output.get('market_impact', {})
        if not isinstance(market_impact, dict):
            errors.append(f"market_impact must be a dictionary")
        else:
            for field in ['stocks', 'bonds', 'currency']:
                if field in market_impact:
                    value = market_impact[field]
                    if value not in self.valid_market_values:
                        errors.append(
                            f"market_impact.{field} must be one of {self.valid_market_values}, got: {value}"
                        )

        is_valid = len(errors) == 0
        return is_valid, errors

    def validate_batch_results(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate entire batch of results.

        Args:
            results_df: DataFrame with parsed sentiment results

        Returns:
            Validation statistics and errors
        """
        total_speeches = len(results_df)
        validation_results = {
            'total_speeches': total_speeches,
            'valid_speeches': 0,
            'invalid_speeches': 0,
            'validation_rate': 0.0,
            'errors_by_speech': {},
            'error_summary': {}
        }

        for idx, row in results_df.iterrows():
            speech_id = row.get('speech_id', f'unknown_{idx}')

            # Reconstruct output dict from DataFrame columns
            output = {
                'hawkish_dovish_score': row.get('hawkish_dovish_score'),
                'topics': {
                    'inflation': row.get('topic_inflation'),
                    'growth': row.get('topic_growth'),
                    'financial_stability': row.get('topic_financial_stability'),
                    'labor_market': row.get('topic_labor_market'),
                    'international': row.get('topic_international')
                },
                'uncertainty': row.get('uncertainty'),
                'forward_guidance_strength': row.get('forward_guidance_strength'),
                'key_sentences': row.get('key_sentences', '').split('|') if pd.notna(row.get('key_sentences')) else [],
                'market_impact': {
                    'stocks': row.get('market_impact_stocks'),
                    'bonds': row.get('market_impact_bonds'),
                    'currency': row.get('market_impact_currency'),
                    'reasoning': row.get('market_impact_reasoning')
                },
                'summary': row.get('summary', '')
            }

            is_valid, errors = self.validate_single_output(output, speech_id)

            if is_valid:
                validation_results['valid_speeches'] += 1
            else:
                validation_results['invalid_speeches'] += 1
                validation_results['errors_by_speech'][speech_id] = errors

                # Count error types
                for error in errors:
                    error_type = error.split(':')[0]
                    if error_type not in validation_results['error_summary']:
                        validation_results['error_summary'][error_type] = 0
                    validation_results['error_summary'][error_type] += 1

        # Calculate validation rate
        if total_speeches > 0:
            validation_results['validation_rate'] = (
                validation_results['valid_speeches'] / total_speeches * 100
            )

        return validation_results

    def print_validation_report(self, validation_results: Dict[str, Any]):
        """Print formatted validation report."""
        print("\n" + "=" * 70)
        print("VALIDATION REPORT")
        print("=" * 70)

        print(f"\nOverall Statistics:")
        print(f"  Total speeches: {validation_results['total_speeches']}")
        print(f"  Valid: {validation_results['valid_speeches']}")
        print(f"  Invalid: {validation_results['invalid_speeches']}")
        print(f"  Validation rate: {validation_results['validation_rate']:.1f}%")

        if validation_results['error_summary']:
            print(f"\nCommon Error Types:")
            for error_type, count in sorted(
                validation_results['error_summary'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                print(f"  - {error_type}: {count} occurrences")

        if validation_results['errors_by_speech']:
            print(f"\nSample of Problematic Speeches (first 5):")
            for i, (speech_id, errors) in enumerate(
                list(validation_results['errors_by_speech'].items())[:5]
            ):
                print(f"\n  Speech: {speech_id}")
                for error in errors[:3]:
                    print(f"    - {error}")

        print("\n" + "=" * 70)

"""
Metric thresholds for automated prompt improvement.

Use MetricThreshold.get_threshold(metric_name) to look up numeric threshold values.
"""

from enum import Enum
from typing import Optional


class MetricThreshold(Enum):
    """Threshold values for each metric by its class name."""

    BLEU = 0.5
    ROUGE_1 = 0.5
    ROUGE_2 = 0.5
    ROUGE_3 = 0.5
    METEOR = 0.5
    GLEU = 0.3
    LEPOR = 0.4
    TER = 0.8  # Higher TER = worse
    COMET = 0.6

    @classmethod
    def get_threshold(cls, metric_name: str) -> Optional[float]:
        """
        Return the threshold for a given metric class name, or None if not found.
        """
        try:
            return cls[metric_name.upper()].value
        except KeyError:
            return None


class CategoryThreshold(Enum):
    """
    Thresholds for LLM judge evaluation categories.
    These correspond to EvalCategory values.
    """

    FACTUALITY = 0.8
    FLUENCY = 0.8
    HELPFULNESS = 0.7
    RELEVANCE = 0.8
    SAFETY = 0.9
    CREATIVITY = 0.6


    @classmethod
    def get_threshold(cls, category_name: str) -> Optional[float]:
        """
        Return the threshold for a given evaluation category name, or None if not set.
        """
        try:
            return cls[category_name.upper()].value
        except KeyError:
            return None

from jiwer import wer
from agent_eval.metrics.base import BaseMetric
from agent_eval.utils.logging_utils import loggable


class TERMetric(BaseMetric):
    """
    TER (Translation Edit Rate) metric using jiwer's Word Error Rate (WER) as a proxy.

    TER measures how many edits (insertions, deletions, substitutions) are needed
    to change the generated text into the reference. This version uses jiwer.wer(),
    which performs tokenization, normalization, and computes edit distance over words.

    Attributes:
        aliases: Alternative names for this metric
        suggestion: Prompt improvement advice when TER is high
    """

    aliases = ["ter", "TER", "ter-metric"]
    suggestion = (
        "If TER is high, the output differs too much from the reference—"
        "improve phrasing and structure so it aligns more closely with the reference."
    )

    def __init__(self, criteria="ter"):
        super().__init__(criteria)

    @loggable
    def evaluate(self, generated, reference=None, prompt=None, user_query=None, **kwargs):
        if reference is None:
            return {"score": None, "error": "Reference required for TER."}
        try:
            score = wer(reference, generated)  # returns a float between 0 and 1
            return {"score": score, "suggestion": self.suggestion}
        except Exception as e:
            return {"score": None, "error": f"TER evaluation failed: {str(e)}"}

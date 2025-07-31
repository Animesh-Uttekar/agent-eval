from agent_eval.metrics.base import BaseMetric
from agent_eval.utils.logging_utils import loggable
import nltk
import math
from collections import Counter


class LEPORMetric(BaseMetric):
    aliases = ["lepor", "lepormetric", "LEPOR"]
    suggestion = (
        "If LEPOR is low, your output needs better overlap in coverage, length, and word order. "
        "Try prompting with explicit phrasing or structure cues."
    )

    def __init__(self, criteria="lepor"):
        super().__init__(criteria)

    @loggable
    def evaluate(self, generated, reference=None, **kwargs):
        if reference is None:
            return {"score": None, "error": "Reference required for LEPOR."}

        try:
            try:
                gtoks = nltk.word_tokenize(generated)
                rtoks = nltk.word_tokenize(reference)
            except LookupError:
                gtoks = generated.split()
                rtoks = reference.split()

            gc = Counter(gtoks)
            rc = Counter(rtoks)
            overlap = sum(min(gc[t], rc[t]) for t in gc if t in rc)
            precision = overlap / len(gtoks) if gtoks else 0.0
            recall = overlap / len(rtoks) if rtoks else 0.0
            if precision + recall == 0:
                f_term = 0.0
            else:
                f_term = 2 * precision * recall / (precision + recall)

            len_g, len_r = len(gtoks), len(rtoks)
            if len_g == 0 or len_r == 0:
                return {"score": 0.0, "suggestion": self.suggestion}
            ratio = len_g / len_r
            lp = math.exp(-abs(math.log(ratio)))

            order_scores = []
            for tok in set(gtoks) & set(rtoks):
                i_g = gtoks.index(tok)
                i_r = rtoks.index(tok)
                order_scores.append(1.0 / (1.0 + abs(i_g - i_r)))
            worp = sum(order_scores) / len(order_scores) if order_scores else 1.0

            score = lp * f_term * worp
            return {"score": score, "suggestion": self.suggestion}
        except Exception as e:
            return {"score": None, "error": f"LEPOR evaluation failed: {e}"}

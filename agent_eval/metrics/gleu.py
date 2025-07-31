from agent_eval.metrics.base import BaseMetric
from agent_eval.utils.logging_utils import loggable
import nltk
from nltk.translate.gleu_score import sentence_gleu


class GLEUMetric(BaseMetric):
    aliases = ["gleu", "gleu_metric", "GLEU"]
    suggestion = "If GLEU is low, your output may not sufficiently match the reference. Improve by adding key phrases and words similar to the reference."

    def __init__(self, criteria="gleu"):
        super().__init__(criteria)

    @loggable
    def evaluate(self, generated, reference=None, **kwargs):
        if reference is None:
            return {"score": None, "error": "Reference required for GLEU."}
        try:
            try:
                ref_tokens = [nltk.word_tokenize(reference)]
                gen_tokens = nltk.word_tokenize(generated)
            except LookupError:
                ref_tokens = [reference.split()]
                gen_tokens = generated.split()

            score = sentence_gleu(ref_tokens, gen_tokens)
            return {"score": score, "suggestion": self.suggestion}
        except Exception as e:
            return {"score": None, "error": f"GLEU evaluation failed: {str(e)}"}

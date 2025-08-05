from math import exp
from agent_eval.metrics.base import BaseMetric
from agent_eval.utils.logging_utils import loggable

try:
    from comet import download_model, load_from_checkpoint
except ImportError:
    raise ImportError("Please install `unbabel-comet` with: pip install unbabel-comet")


class COMETMetric(BaseMetric):
    """
    COMET (Crosslingual Optimized Metric for Evaluation of Translation) is a multilingual
    metric trained on human-labeled data to assess adequacy and fluency in translations
    or text generation tasks.

    COMET uses pre-trained models fine-tuned on quality estimation tasks and requires
    the source input, machine translation (generated), and reference text.

    Attributes:
        aliases: Alternative names for this metric.
        suggestion: Advice to improve prompt if COMET score is low.
    """

    aliases = ["comet", "comet-metric", "COMET"]
    suggestion = "If COMET is low, try adding clearer instruction like 'translate meaningfully and preserve nuance' to ensure both adequacy and fluency."

    def __init__(self, criteria="comet"):
        super().__init__(criteria)
        model_path = download_model("Unbabel/wmt20-comet-da")
        self.model = load_from_checkpoint(model_path)

    def _normalize_score(self, score):
        return 1 / (1 + exp(-score))

    @loggable
    def evaluate(
        self, generated, reference=None, prompt=None, user_query=None, **kwargs
    ):
        """
        Evaluate the generated text using the COMET metric.

        Args:
            generated (str): The generated model output.
            reference (str): The reference output text.
            input_text (str): The original source input (optional but recommended for COMET).
            **kwargs: Extra arguments (ignored).

        Returns:
            dict: {
                'score': float or None,
                'suggestion': str,
                'error': str (optional)
            }

        Example:
            >>> comet = COMETMetric()
            >>> comet.evaluate("Le chat dort.", "The cat is sleeping.", input_text="The cat sleeps.")
        """
        if reference is None:
            return {"score": None, "error": "Reference required for COMET."}
        if user_query is None:
            return {"score": None, "error": "User Query required for COMET."}

        try:
            data = [{"src": user_query, "mt": generated, "ref": reference}]
            raw_result = self.model.predict(data, batch_size=8, gpus=1)

            raw_score = raw_result[0] if isinstance(raw_result, list) else raw_result
            if hasattr(raw_score, "scores"):
                score_val = raw_score.scores[0]
            elif isinstance(raw_score, dict) and "scores" in raw_score:
                score_val = raw_score["scores"][0]
            else:
                return {"score": None, "error": "COMET score missing in response."}

            normalized = self._normalize_score(score_val)

            return {"score": normalized, "suggestion": self.suggestion}

        except Exception as e:
            return {"score": None, "error": f"COMET evaluation failed: {str(e)}"}

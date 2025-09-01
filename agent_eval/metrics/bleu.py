from agent_eval.metrics.base import BaseMetric
from agent_eval.utils.logging_utils import loggable
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class BLEUMetric(BaseMetric):
    """
    BLEU (Bilingual Evaluation Understudy) metric for evaluating text generation quality.

    BLEU measures n-gram precision between generated and reference text, with brevity penalty.
    It is widely used in machine translation and text generation evaluation.

    The metric calculates the geometric mean of n-gram precision scores (typically for n=1,2,3,4)
    and applies a brevity penalty to penalize overly short translations.

    Attributes:
        aliases: Alternative names for this metric including 'bleu', 'bleu-metric', 'BLEU'
        suggestion: Prompt improvement advice when BLEU score is low
    """

    aliases = ["bleu", "bleu_metric", "BLEU", "bleu-metric"]
    suggestion = "If BLEU is low, make your prompt more specific and include key phrases from the reference so the model's output aligns more closely with the reference wording."

    def __init__(self, criteria="bleu"):
        super().__init__(criteria)

    @loggable
    def evaluate(self, generated, reference=None, prompt=None, user_query=None, **kwargs):
        """
        Evaluate generated text against reference using BLEU score.

        This method tokenizes both the generated and reference text, then calculates
        the BLEU score using NLTK's implementation with smoothing to handle edge cases.

        Args:
            generated (str): The generated text to be evaluated
            reference (str, optional): The reference text to compare against. Required for BLEU.
            **kwargs: Additional keyword arguments (not used in BLEU calculation)

        Returns:
            dict: A dictionary containing:
                - 'score' (float): BLEU score between 0 and 1, or None if error
                - 'suggestion' (str): Prompt improvement advice
                - 'error' (str, optional): Error message if evaluation fails

        Raises:
            Exception: If tokenization or BLEU calculation fails

        Example:
            >>> metric = BLEUMetric()
            >>> result = metric.evaluate("A quick brown fox", "The quick brown fox jumps")
            >>> print(result['score'])
            0.5
        """
        if reference is None:
            return {"score": None, "error": "Reference required for BLEU."}
        try:
            try:
                ref_tokens = [nltk.word_tokenize(reference)]
                gen_tokens = nltk.word_tokenize(generated)
            except LookupError:
                ref_tokens = [reference.split()]
                gen_tokens = generated.split()
            score = sentence_bleu(
                ref_tokens, gen_tokens, smoothing_function=SmoothingFunction().method1
            )
            return {"score": score, "suggestion": self.suggestion}
        except Exception as e:
            return {"score": None, "error": f"BLEU evaluation failed: {str(e)}"}

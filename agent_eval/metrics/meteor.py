from agent_eval.metrics.base import BaseMetric
from agent_eval.utils.logging_utils import loggable
import nltk


class METEORMetric(BaseMetric):
    """
    METEOR (Metric for Evaluation of Translation with Explicit ORdering) metric.

    METEOR is an evaluation metric that addresses some of the limitations of BLEU.
    It uses word alignment between the generated and reference text, considering
    exact matches, stemmed matches, and synonym matches.

    The metric calculates precision, recall, and F-score based on word alignments,
    and includes a penalty for word order differences.

    Attributes:
        aliases: Alternative names for this metric including 'meteor', 'meteor-metric', 'METEOR'
        suggestion: Prompt improvement advice when METEOR score is low
    """

    aliases = ["meteor", "meteor_metric", "METEOR", "meteor-metric"]
    suggestion = "If METEOR is low, the model may not be capturing semantically equivalent terms. Try adding guidance in your prompt to use synonyms or rephrase content in ways that stay true to the reference meaning, improving word alignment."

    def __init__(self, criteria="meteor"):
        super().__init__(criteria)

    @loggable
    def evaluate(self, generated, reference=None, prompt=None, user_query=None, **kwargs):
        """
        Evaluate generated text against reference using METEOR score.

        This method tokenizes both the generated and reference text, then calculates
        the METEOR score using NLTK's implementation which considers exact matches,
        stemmed matches, and synonym matches.

        Args:
            generated (str): The generated text to be evaluated
            reference (str, optional): The reference text to compare against. Required for METEOR.
            **kwargs: Additional keyword arguments (not used in METEOR calculation)

        Returns:
            dict: A dictionary containing:
                - 'score' (float): METEOR score between 0 and 1, or None if error
                - 'suggestion' (str): Prompt improvement advice
                - 'error' (str, optional): Error message if evaluation fails

        Raises:
            Exception: If tokenization or METEOR calculation fails

        Example:
            >>> metric = METEORMetric()
            >>> result = metric.evaluate("A quick brown fox", "The quick brown fox jumps")
            >>> print(result['score'])
            0.6
        """
        if reference is None:
            return {"score": None, "error": "Reference required for METEOR."}
        try:
            try:
                ref_tokens = nltk.word_tokenize(reference)
                gen_tokens = nltk.word_tokenize(generated)
            except LookupError:
                ref_tokens = reference.split()
                gen_tokens = generated.split()
            score = nltk.translate.meteor_score.meteor_score([ref_tokens], gen_tokens)
            return {"score": score, "suggestion": self.suggestion}
        except Exception as e:
            return {"score": None, "error": f"METEOR evaluation failed: {str(e)}"}

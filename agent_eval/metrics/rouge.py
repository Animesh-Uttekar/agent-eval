from agent_eval.metrics.base import BaseMetric
from rouge_score import rouge_scorer
from agent_eval.utils.logging_utils import loggable


class ROUGE1Metric(BaseMetric):
    """
    ROUGE-1 metric for evaluating text generation quality.

    ROUGE-1 measures unigram (single word) recall between generated and reference text.
    It calculates the proportion of words from the reference that appear in the generated text.

    This metric is particularly useful for evaluating summarization tasks where
    capturing key words from the source is important.

    Attributes:
        aliases: Alternative names for this metric including 'rouge-1', 'rouge1', 'ROUGE-1'
        suggestion: Prompt improvement advice when ROUGE-1 score is low
    """

    aliases = [
        "rouge1",
        "rouge_1",
        "rouge-1",
        "ROUGE_1",
        "rouge_1_metric",
        "rouge-1-metric",
    ]
    suggestion = "If ROUGE-1 is low, the output is missing important keywords from the input/reference. Emphasize those keywords or critical facts in your prompt to ensure they appear in the model's response."

    def __init__(self, criteria="rouge1"):
        super().__init__(criteria)

    @loggable
    def evaluate(self, generated, reference=None, **kwargs):
        """
        Evaluate generated text against reference using ROUGE-1 score.

        This method calculates the unigram recall score using the rouge-score library,
        which measures the proportion of reference words that appear in the generated text.

        Args:
            generated (str): The generated text to be evaluated
            reference (str, optional): The reference text to compare against. Required for ROUGE-1.
            **kwargs: Additional keyword arguments (not used in ROUGE-1 calculation)

        Returns:
            dict: A dictionary containing:
                - 'score' (float): ROUGE-1 recall score between 0 and 1, or None if error
                - 'suggestion' (str): Prompt improvement advice
                - 'error' (str, optional): Error message if evaluation fails

        Example:
            >>> metric = ROUGE1Metric()
            >>> result = metric.evaluate("A quick brown fox", "The quick brown fox jumps")
            >>> print(result['score'])
            0.75
        """
        if reference is None:
            return {"score": None, "error": "Reference required for ROUGE-1."}
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        score = scorer.score(reference, generated)["rouge1"].recall
        return {"score": score, "suggestion": self.suggestion}


class ROUGE2Metric(BaseMetric):
    """
    ROUGE-2 metric for evaluating text generation quality.

    ROUGE-2 measures bigram (two-word phrase) recall between generated and reference text.
    It calculates the proportion of two-word phrases from the reference that appear in the generated text.

    This metric is useful for evaluating how well the generated text captures
    important phrases and word combinations from the reference.

    Attributes:
        aliases: Alternative names for this metric including 'rouge-2', 'rouge2', 'ROUGE-2'
        suggestion: Prompt improvement advice when ROUGE-2 score is low
    """

    aliases = [
        "rouge2",
        "rouge_2",
        "rouge-2",
        "ROUGE_2",
        "rouge_2_metric",
        "rouge-2-metric",
    ]
    suggestion = "If ROUGE-2 is low, the response lacks important two-word phrases from the reference. You can improve this by providing more context or example phrases in the prompt so the model uses similar phrasing."

    def __init__(self, criteria="rouge2"):
        super().__init__(criteria)

    @loggable
    def evaluate(self, generated, reference=None, **kwargs):
        """
        Evaluate generated text against reference using ROUGE-2 score.

        This method calculates the bigram recall score using the rouge-score library,
        which measures the proportion of reference bigrams that appear in the generated text.

        Args:
            generated (str): The generated text to be evaluated
            reference (str, optional): The reference text to compare against. Required for ROUGE-2.
            **kwargs: Additional keyword arguments (not used in ROUGE-2 calculation)

        Returns:
            dict: A dictionary containing:
                - 'score' (float): ROUGE-2 recall score between 0 and 1, or None if error
                - 'suggestion' (str): Prompt improvement advice
                - 'error' (str, optional): Error message if evaluation fails

        Example:
            >>> metric = ROUGE2Metric()
            >>> result = metric.evaluate("A quick brown fox", "The quick brown fox jumps")
            >>> print(result['score'])
            0.5
        """
        if reference is None:
            return {"score": None, "error": "Reference required for ROUGE-2."}
        scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)
        score = scorer.score(reference, generated)["rouge2"].recall
        return {"score": score, "suggestion": self.suggestion}


class ROUGELMetric(BaseMetric):
    """
    ROUGE-L metric for evaluating text generation quality.

    ROUGE-L measures the longest common subsequence (LCS) between generated and reference text.
    It calculates the proportion of the longest matching sequence of words that appears
    in both the reference and generated text in the same order.

    This metric is particularly useful for evaluating how well the generated text
    maintains the structure and flow of the reference text.

    Attributes:
        aliases: Alternative names for this metric including 'rouge-l', 'rougel', 'ROUGE-L'
        suggestion: Prompt improvement advice when ROUGE-L score is low
    """

    aliases = [
        "rougeL",
        "rouge_l",
        "rouge-L",
        "rougel",
        "ROUGE_L",
        "rouge_l_metric",
        "ROUGE-L",
    ]
    suggestion = "If ROUGE-L is low, the output isn't capturing longer sequences of the reference. Consider instructing the model to follow the structure or wording of the source more closely, ensuring it includes the main sequences of information."

    def __init__(self, criteria="rougeL"):
        super().__init__(criteria)

    @loggable
    def evaluate(self, generated, reference=None, **kwargs):
        """
        Evaluate generated text against reference using ROUGE-L score.

        This method calculates the longest common subsequence recall score using the rouge-score library,
        which measures the proportion of the longest matching sequence of words that appears
        in both the reference and generated text in the same order.

        Args:
            generated (str): The generated text to be evaluated
            reference (str, optional): The reference text to compare against. Required for ROUGE-L.
            **kwargs: Additional keyword arguments (not used in ROUGE-L calculation)

        Returns:
            dict: A dictionary containing:
                - 'score' (float): ROUGE-L recall score between 0 and 1, or None if error
                - 'suggestion' (str): Prompt improvement advice
                - 'error' (str, optional): Error message if evaluation fails

        Example:
            >>> metric = ROUGELMetric()
            >>> result = metric.evaluate("A quick brown fox", "The quick brown fox jumps")
            >>> print(result['score'])
            0.6
        """
        if reference is None:
            return {"score": None, "error": "Reference required for ROUGE-L."}
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        score = scorer.score(reference, generated)["rougeL"].recall
        return {"score": score, "suggestion": self.suggestion}

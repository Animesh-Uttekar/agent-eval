from agent_eval.metrics.base import BaseMetric
from agent_eval.utils.logging_utils import loggable

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


class BLEURTMetric(BaseMetric):
    """
    BLEURT (Bilingual Evaluation Understudy with Representations from Transformers)
    is a learned evaluation metric fine-tuned on human annotations.

    It estimates the quality of a generated output based on how close it is to human preferences,
    in terms of both semantics and fluency.

    BLEURT is more sensitive to subtle nuances than traditional n-gram based metrics.

    Attributes:
        aliases: List of alternative names for BLEURT
        suggestion: Prompt improvement tip for low BLEURT scores
    """

    aliases = ["bleurt", "bleurt_metric", "BLEURT"]
    suggestion = (
        "If BLEURT is low, refine your prompt to encourage natural, context-appropriate answers."
        " For example, include an example of a good answer."
    )

    def __init__(self, criteria="bleurt", model_name="Elron/bleurt-base-512"):
        super().__init__(criteria)
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

    @loggable
    def evaluate(self, generated: str, reference: str = None, **kwargs):
        if reference is None:
            return {"score": None, "error": "Reference required for BLEURT."}
        try:
            inputs = self.tokenizer(
                {
                    "text": reference,
                    "text_pair": generated
                },
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            
            with torch.no_grad():
                scores = self.model(**inputs).logits.squeeze().cpu().numpy()
                score = float(scores.item()) if scores.ndim == 0 else float(np.mean(scores))

            return {
                "score": round(score, 4),
                "suggestion": self.suggestion,
            }

        except Exception as e:
            return {
                "score": None,
                "error": f"BLEURT evaluation failed: {str(e)}",
            }
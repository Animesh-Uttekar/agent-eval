import os
from openai import OpenAI
from agent_eval.core.evaluator import Evaluator
from dotenv import load_dotenv

load_dotenv()


class TestRougeMetric:
    def setup_class(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.evaluator = Evaluator(model=self.client, model_name="gpt-3.5-turbo")

    def test_rouge1_high_similarity(self):
        result = self.evaluator.evaluate(
            prompt="Summarize: Cats are independent animals.",
            model_output="Cats are very independent pets.",
            reference_output="Cats are independent animals.",
            metrics=["rouge1"],
        )
        assert result["metrics"]["rouge1"]["score"] > 0.6

    def test_rouge1_low_similarity(self):
        result = self.evaluator.evaluate(
            prompt="Summarize: Cats are independent animals.",
            model_output="Dogs are loyal companions.",
            reference_output="Cats are independent animals.",
            metrics=["rouge1"],
        )
        assert result["metrics"]["rouge1"]["score"] < 0.3

    def test_rouge2_high_similarity(self):
        result = self.evaluator.evaluate(
            prompt="Paraphrase: The sun heats the earth.",
            model_output="The sun warms the planet.",
            reference_output="The sun heats the earth.",
            metrics=["rouge2"],
        )
        assert result["metrics"]["rouge2"]["score"] > 0.2

    def test_rouge2_low_similarity(self):
        result = self.evaluator.evaluate(
            prompt="Paraphrase: The sun heats the earth.",
            model_output="Birds fly in the sky.",
            reference_output="The sun heats the earth.",
            metrics=["rouge2"],
        )
        assert result["metrics"]["rouge2"]["score"] < 0.2

    def test_rougeL_high_similarity(self):
        result = self.evaluator.evaluate(
            prompt="Explain rain formation.",
            model_output="Rain forms when water vapor condenses into droplets and falls.",
            reference_output="Rain is formed by condensed water vapor that falls as droplets.",
            metrics=["rougeL"],
        )
        assert result["metrics"]["rougeL"]["score"] > 0.4

    def test_rougeL_low_similarity(self):
        result = self.evaluator.evaluate(
            prompt="Explain rain formation.",
            model_output="Photosynthesis helps plants grow.",
            reference_output="Rain is formed by condensed water vapor that falls as droplets.",
            metrics=["rougeL"],
        )
        assert result["metrics"]["rougeL"]["score"] < 0.2

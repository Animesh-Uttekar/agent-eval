import os
from openai import OpenAI
from agent_eval.core.evaluator import Evaluator
from dotenv import load_dotenv

load_dotenv()


class TestMeteorMetric:
    def setup_class(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.evaluator = Evaluator(model=self.client, model_name="gpt-3.5-turbo")

    def test_meteor_high_similarity(self):
        result = self.evaluator.evaluate(
            prompt="Define ecosystem.",
            model_output="An ecosystem is a community of living organisms and their environment.",
            reference_output="An ecosystem includes living organisms and their environment.",
            metrics=["meteor"],
        )
        assert result["metrics"]["meteor"]["score"] > 0.6

    def test_meteor_low_similarity(self):
        result = self.evaluator.evaluate(
            prompt="Define ecosystem.",
            model_output="Gravity causes objects to fall.",
            reference_output="An ecosystem includes living organisms and their environment.",
            metrics=["meteor"],
        )
        assert result["metrics"]["meteor"]["score"] < 0.3

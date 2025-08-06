import os
from openai import OpenAI
from agent_eval.core.evaluator import Evaluator
from dotenv import load_dotenv

load_dotenv()


class TestBLEURTMetric:
    def setup_class(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.evaluator = Evaluator(model=self.client, model_name="gpt-3.5-turbo")

    def test_bleurt_high_similarity(self):
        result = self.evaluator.evaluate(
            model_output="The water cycle involves evaporation, condensation, precipitation, and collection.",
            reference_output="Evaporation, condensation, precipitation, and collection are parts of the water cycle.",
            prompt="Explain the water cycle.",
            metrics=["bleurt"],
        )
        assert result["metrics"]["bleurt"]["score"] > 0.4

    def test_bleurt_low_similarity(self):
        result = self.evaluator.evaluate(
            model_output="The process of photosynthesis converts sunlight into chemical energy in plants.",
            reference_output="Evaporation, condensation, precipitation, and collection are parts of the water cycle.",
            prompt="Explain the water cycle.",
            metrics=["bleurt"],
        )
        assert result["metrics"]["bleurt"]["score"] < 0.3
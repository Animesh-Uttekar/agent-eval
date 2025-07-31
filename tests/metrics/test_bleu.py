import os
from openai import OpenAI
from agent_eval.core.evaluator import Evaluator
from dotenv import load_dotenv

load_dotenv()


class TestBLEUMetric:
    def setup_class(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.evaluator = Evaluator(model=self.client, model_name="gpt-3.5-turbo")

    def test_bleu_high_similarity(self):
        result = self.evaluator.evaluate(
            prompt="Explain the water cycle.",
            model_output="The water cycle includes evaporation, condensation, precipitation, and collection.",
            reference_output="Evaporation, condensation, precipitation, and collection are parts of the water cycle.",
            metrics=["bleu"],
        )
        assert result["metrics"]["bleu"]["score"] > 0.4

    def test_bleu_low_similarity(self):
        result = self.evaluator.evaluate(
            prompt="Explain the water cycle.",
            model_output="Photosynthesis is a process used by plants to convert light into energy.",
            reference_output="Evaporation, condensation, precipitation, and collection are parts of the water cycle.",
            metrics=["bleu"],
        )
        assert result["metrics"]["bleu"]["score"] < 0.2

import os
from openai import OpenAI
from agent_eval.core.evaluator import Evaluator
from multiprocessing import get_logger
from dotenv import load_dotenv

load_dotenv()

logger = get_logger()


class TestTERMetric:
    def setup_class(self):
        logger.info("Setting up OpenAI client with gpt-3.5-turbo...")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.evaluator = Evaluator(model=self.client, model_name="gpt-3.5-turbo")
        logger.info("Client setup complete.\n")

    def test_ter_high_similarity(self):
        prompt = "Translate: Bonjour"
        model_output = "Hello"
        reference_output = "Hello"
        result = self.evaluator.evaluate(
            prompt=prompt,
            model_output=model_output,
            reference_output=reference_output,
            metrics=["ter"],
        )
        assert result["metrics"]["ter"]["score"] < 0.2

    def test_ter_low_similarity(self):
        prompt = "Translate: Bonjour"
        model_output = "Goodbye"
        reference_output = "Hello"
        result = self.evaluator.evaluate(
            prompt=prompt,
            model_output=model_output,
            reference_output=reference_output,
            metrics=["ter"],
        )
        assert result["metrics"]["ter"]["score"] > 0.9

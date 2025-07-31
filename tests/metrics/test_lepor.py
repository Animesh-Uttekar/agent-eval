import os
from openai import OpenAI
from agent_eval.core.evaluator import Evaluator
from multiprocessing import get_logger
from dotenv import load_dotenv

load_dotenv()

logger = get_logger()


class TestLEPORMetric:
    def setup_class(self):
        logger.info("Setting up OpenAI client with gpt-3.5-turbo...")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.evaluator = Evaluator(model=self.client, model_name="gpt-3.5-turbo")
        logger.info("Client setup complete.\n")

    def test_lepor_high_score(self):
        prompt = "Summarize: The sun evaporates water into the sky"
        model_output = "The sun causes water to evaporate into the sky"
        reference_output = "The sun evaporates water into the sky"
        result = self.evaluator.evaluate(
            prompt=prompt,
            model_output=model_output,
            reference_output=reference_output,
            metrics=["lepor"],
        )
        assert result["metrics"]["lepor"]["score"] > 0.35

    def test_lepor_low_score(self):
        prompt = "Summarize: The sun evaporates water into the sky"
        model_output = "Birds fly south during winter"
        reference_output = "The sun evaporates water into the sky"
        result = self.evaluator.evaluate(
            prompt=prompt,
            model_output=model_output,
            reference_output=reference_output,
            metrics=["lepor"],
        )
        assert result["metrics"]["lepor"]["score"] < 0.2

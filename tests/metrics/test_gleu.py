from multiprocessing import get_logger
import os
from openai import OpenAI
from agent_eval.core.evaluator import Evaluator
from dotenv import load_dotenv

load_dotenv()

logger = get_logger()


class TestGLEUMetric:
    def setup_class(self):
        logger.info("Setting up OpenAI client with gpt-3.5-turbo...")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.evaluator = Evaluator(model=self.client, model_name="gpt-3.5-turbo")
        logger.info("Client setup complete.\n")

    def test_gleu_good_overlap(self):
        prompt = "Paraphrase this: The cat is sleeping"
        model_output = "The cat is taking a nap"
        reference_output = "The cat is sleeping"
        result = self.evaluator.evaluate(
            prompt=prompt,
            model_output=model_output,
            reference_output=reference_output,
            metrics=["gleu"],
        )
        assert result["metrics"]["gleu"]["score"] >= 0.3

    def test_gleu_bad_overlap(self):
        prompt = "Paraphrase this: The cat is sleeping"
        model_output = "He went to the store"
        reference_output = "The cat is sleeping"
        result = self.evaluator.evaluate(
            prompt=prompt,
            model_output=model_output,
            reference_output=reference_output,
            metrics=["gleu"],
        )
        assert result["metrics"]["gleu"]["score"] <= 0.2

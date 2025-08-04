from multiprocessing import get_logger
import os
from openai import OpenAI
from agent_eval.core.evaluator import Evaluator
from dotenv import load_dotenv

load_dotenv()
logger = get_logger()


class TestCOMETMetric:
    def setup_class(self):
        logger.info("Setting up OpenAI client with gpt-3.5-turbo...")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.evaluator = Evaluator(model=self.client, model_name="gpt-3.5-turbo")
        logger.info("Client setup complete.\n")

    def test_translation_agent(self):
        prompt = "Translate the sentence to German"
        model_output = "Das Wetter ist heute schön und ich gehe spazieren."
        reference_output = "Das Wetter ist heute schön, und ich mache einen Spaziergang."
        user_query = "The weather is nice today, and I'm going for a walk."
        result = self.evaluator.evaluate(
            prompt=prompt,
            model_output=model_output,
            reference_output=reference_output,
            user_query=user_query,
            metrics=["comet"],
            prompt_optimizer=True
        )
        assert result["metrics"]["comet"]["score"] >= 0.6

    def test_customer_support_agent_good(self):
        prompt = "As a customer service AI, respond helpfully"
        model_output = "I'm sorry to hear that. Let me run a quick check on your connection. Could you confirm your account number?"
        reference_output = "Sorry for the inconvenience. Please share your account ID so I can assist you further."
        user_query = "My internet is not working since morning. Can you help?"
        result = self.evaluator.evaluate(
            prompt=prompt,
            model_output=model_output,
            reference_output=reference_output,
            user_query=user_query,
            metrics=["comet"],
            prompt_optimizer=True,
            max_prompt_improvements=2
        )
        assert result["metrics"]["comet"]["score"] >= 0.4

    def test_customer_support_agent_bad(self):
        prompt = "As a customer service AI, respond helpfully"
        model_output = "Try buying a new router. That's all I can say."
        reference_output = "Sorry for the inconvenience. Please share your account ID so I can assist you further."
        user_query = "My internet is not working since morning. Can you help?"
        result = self.evaluator.evaluate(
            prompt=prompt,
            model_output=model_output,
            reference_output=reference_output,
            user_query=user_query,
            metrics=["comet"],
            prompt_optimizer=True
        )
        assert result["metrics"]["comet"]["score"] <= 0.3
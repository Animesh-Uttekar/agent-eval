import os
from typing import Optional
from dotenv import load_dotenv
from multiprocessing import get_logger
from openai import OpenAI
from agent_eval.core.evaluator import Evaluator
from agent_eval.utils.logging_utils import loggable

load_dotenv()

logger = get_logger()


class TestLLMJudgeReferenceOnly:
    def setup_class(self):

        logger.info("Setting up OpenAI client with gpt-3.5-turbo...")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.evaluator = Evaluator(model=self.client, model_name="gpt-3.5-turbo")
        logger.info("Client setup complete.\n")

    @loggable
    def _evaluate_single_judge(
        self, criteria, prompt, model_output: Optional[str] = None, reference_output=""
    ):
        logger.info("Evaluating model...")

        return self.evaluator.evaluate(
            prompt=prompt,
            model_output=model_output,
            reference_output=reference_output,
            judges=criteria,
            prompt_optimizer=True,
            max_prompt_improvements=2
        )

    def test_fluency_minor_errors(self):
        logger.info("Running test: Fluency (Minor Errors)")
        result = self._evaluate_single_judge(
            "fluency",
            prompt="Explain how planes fly.",
            model_output="Planes flies using lift which is created by wings as air goes over them.",
        )
        logger.info("[Fluency Minor Errors Result]", result)

        assert 0.4 <= result["judges"]["fluency"]["score"] <= 0.9
        logger.info("Minor fluency error test passed.\n")

    def test_fluency_awkward_phrasing(self):
        logger.info("Running test: Fluency (Awkward Phrasing)")
        result = self._evaluate_single_judge(
            "fluency",
            prompt="Describe the function of the heart.",
            model_output="The heart is working for pumping blood that goes in the body and all organs have blood from that.",
        )
        logger.info("[Fluency Awkward Phrasing Result]", result)
        assert 0.3 <= result["judges"]["fluency"]["score"] <= 0.7
        logger.info("Awkward fluency test passed.\n")

    def test_fluency_with_repetition(self):
        logger.info("Running test: Fluency (Repetitive)")
        result = self._evaluate_single_judge(
            "fluency",
            prompt="Why is exercise important?",
            model_output="Exercise is important because exercise helps you stay healthy. Exercise helps the body. Exercise also makes you feel good. Exercise is good.",
        )
        logger.info("[Fluency Repetitive Result]", result)
        assert result["judges"]["fluency"]["score"] <= 0.5
        logger.info("Repetitive fluency test passed.\n")

    def test_fluency_with_fragmented_sentences(self):
        logger.info("Running test: Fluency (Fragmented)")
        result = self._evaluate_single_judge(
            "fluency",
            prompt="What is gravity?",
            model_output="Gravity. Pulls everything. Towards the Earth. Keeps moon. In orbit.",
        )
        logger.info("[Fluency Fragmented Result]", result)
        assert result["judges"]["fluency"]["score"] <= 0.4
        logger.info("Fragmented fluency test passed.\n")

    def test_fluency_incoherent(self):
        logger.info("Running test: Fluency (Incoherent)")
        result = self._evaluate_single_judge(
            "fluency",
            prompt="What is global warming?",
            model_output="Global warming it go hot, the earth more, is making melting it, because that car smoke elephant fish ocean hot.",
        )
        logger.info("[Fluency Incoherent Result]", result)
        assert result["judges"]["fluency"]["score"] <= 0.2
        logger.info("Incoherent fluency test passed.\n")

    def test_relevance_judge_with_reference(self):
        logger.info("Running test: Relevance")
        result = self._evaluate_single_judge(
            "relevance",
            prompt="Describe the Eiffel Tower.",
            model_output="It is a tower located in Paris, France.",
            reference_output="The Eiffel Tower is a famous iron structure located in Paris, France, built in 1889 as the entrance to the World's Fair.",
        )
        logger.info("[Relevance Result]", result)
        assert isinstance(result["judges"]["relevance"]["score"], (int, float))
        logger.info("Relevance test passed.\n")

    def test_relevance_irrelevant_context(self):
        logger.info("Running test: Relevance (Irrelevant Context)")
        result = self._evaluate_single_judge(
            "relevance",
            prompt="Describe the process of mitosis.",
            model_output="The Eiffel Tower is located in Paris and is made of iron.",
            reference_output="Mitosis is a type of cell division that results in two daughter cells with the same number of chromosomes as the parent.",
        )
        logger.info("[Relevance Irrelevant Result]", result)
        logger.info("Irrelevant relevance test passed.\n")

    def test_helpfulness_partial_response(self):
        logger.info("Running test: Helpfulness (Partial Response)")
        result = self._evaluate_single_judge(
            "helpfulness",
            prompt="Explain how photosynthesis works.",
            model_output="Photosynthesis involves plants converting sunlight into energy.",
            reference_output="Photosynthesis is the process where green plants use sunlight, carbon dioxide, and water to produce glucose and oxygen.",
        )
        logger.info("[Helpfulness Partial Result]", result)
        logger.info("Partial helpfulness test passed.\n")

    def test_helpfulness_judge_with_reference(self):
        logger.info("Running test: Helpfulness")
        result = self._evaluate_single_judge(
            "helpfulness",
            prompt="Explain what a neural network is.",
            model_output="A neural network is a system of algorithms that attempts to recognize patterns in data.",
            reference_output="A neural network is a machine learning model inspired by the human brain that consists of layers of nodes (neurons) and is used to recognize patterns in data.",
        )
        logger.info("[Helpfulness Result]", result)
        assert isinstance(result["judges"]["helpfulness"]["score"], (int, float))
        logger.info("Helpfulness test passed.\n")

    def test_helpfulness_incorrect_response(self):
        logger.info("Running test: Helpfulness (Incorrect Response)")
        result = self._evaluate_single_judge(
            "helpfulness",
            prompt="What causes tides in the ocean?",
            model_output="Tides are caused by underwater volcanoes.",
            reference_output="Tides are primarily caused by the gravitational pull of the moon and the sun on Earth's oceans.",
        )
        logger.info("[Helpfulness Incorrect Result]", result)
        logger.info("Incorrect helpfulness test passed.\n")

    def test_relevance_with_partial_context(self):
        logger.info("Running test: Relevance (Partial Context)")
        result = self._evaluate_single_judge(
            "relevance",
            prompt="What are the benefits of regular exercise?",
            model_output="Exercise can improve cardiovascular health and help with weight management.",
            reference_output="Exercise improves physical and mental health, including cardiovascular fitness, strength, flexibility, and mood.",
        )
        logger.info("[Relevance Partial Result]", result)
        logger.info("Partial relevance test passed.\n")

    def test_factuality_perfect_match(self):
        logger.info("Running test: Factuality (Correct Answer)")
        result = self._evaluate_single_judge(
            "factuality",
            prompt="What is the boiling point of water at sea level?",
            model_output="The boiling point of water at sea level is 100 degrees Celsius.",
            reference_output="The boiling point of water at sea level is 100 degrees Celsius (212 degrees Fahrenheit).",
        )
        assert result["judges"]["factuality"]["score"] >= 0.6
        logger.info("Correct factuality test passed.\n")

    def test_factuality_minor_error(self):
        logger.info("Running test: Factuality (Minor Error)")
        result = self._evaluate_single_judge(
            "factuality",
            prompt="Who painted the Mona Lisa?",
            model_output="The Mona Lisa was painted by Michelangelo.",
            reference_output="The Mona Lisa was painted by Leonardo da Vinci.",
        )
        assert result["judges"]["factuality"]["score"] < 0.5
        logger.info("Minor error factuality test passed.\n")

    def test_factuality_judge_with_reference(self):
        logger.info("Running test: Factuality")
        result = self._evaluate_single_judge(
            "factuality",
            prompt="What is the capital of Italy?",
            model_output="The capital of Italy is Venice.",
            reference_output="The capital of Italy is Rome.",
        )
        logger.info("[Factuality Result]", result)
        assert isinstance(result["judges"]["factuality"]["score"], (int, float))
        logger.info("Factuality test passed.\n")

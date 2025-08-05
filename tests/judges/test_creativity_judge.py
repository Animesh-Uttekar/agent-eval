import os
from typing import Optional
from dotenv import load_dotenv
from multiprocessing import get_logger
from openai import OpenAI
from agent_eval.core.evaluator import Evaluator
from agent_eval.utils.logging_utils import loggable

load_dotenv()
logger = get_logger()


class TestCreativityJudgeOnly:
    def setup_class(self):
        logger.info("Setting up OpenAI client with gpt-3.5-turbo...")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.evaluator = Evaluator(model=self.client, model_name="gpt-3.5-turbo")
        logger.info("Client setup complete.\n")

    @loggable
    def _evaluate_single_judge(
        self, criteria, prompt, model_output: str, reference_output: str = ""
    ):
        logger.info(f"Evaluating model with {criteria} judge...")
        return self.evaluator.evaluate(
            prompt=prompt,
            model_output=model_output,
            reference_output=reference_output,
            judges=criteria,
        )

    def test_creativity_poetic_response(self):
        result = self._evaluate_single_judge(
            "creativity",
            prompt="Write a haiku about a warrior facing defeat.",
            model_output="Blade chipped, eyes weary, / Winds mourn battles never won / Silence takes the field.",
        )
        assert result["judges"]["creativity"]["score"] >= 0.8

    def test_creativity_flat_story(self):
        result = self._evaluate_single_judge(
            "creativity",
            prompt="Tell a funny story about a cat who wants to become president.",
            model_output="A cat wanted to be president. It ran for office. It gave speeches. People voted. The cat won.",
        )
        assert result["judges"]["creativity"]["score"] <= 0.5

    def test_creativity_fairy_tale(self):
        result = self._evaluate_single_judge(
            "creativity",
            prompt="Tell a short fairy tale about a blind dragon.",
            model_output="In a mountain veiled by mist, a blind dragon sculpted dreams from smoke. Villagers feared him, until one girl heard his sorrow, and taught him how to see through song.",
        )
        assert result["judges"]["creativity"]["score"] >= 0.85

    def test_creativity_style_violation(self):
        result = self._evaluate_single_judge(
            "creativity",
            prompt="Write a formal obituary for a goldfish named Bubbles.",
            model_output="Yo, Bubbles the fish kicked the bucket. Lived in a bowl, ate flakes. Peace out, gills.",
        )
        assert result["judges"]["creativity"]["score"] <= 0.5

    def test_creativity_imaginative_dialogue(self):
        result = self._evaluate_single_judge(
            "creativity",
            prompt="Write a dialogue between a tree and the wind.",
            model_output='"You tickle my leaves," said the tree. "I carry your secrets," whispered the wind.',
        )
        assert result["judges"]["creativity"]["score"] >= 0.6

    def test_creativity_blend_genres(self):
        result = self._evaluate_single_judge(
            "creativity",
            prompt="Write a noir detective story set in Candyland.",
            model_output="She walked into my sugar shack like a sour patch—sweet with a twist. The jellybean mayor had vanished, and only one man could solve it: me, Detective Licorice.",
        )
        assert result["judges"]["creativity"]["score"] >= 0.6

    def test_creativity_repetitive_text(self):
        result = self._evaluate_single_judge(
            "creativity",
            prompt="Tell a bedtime story for kids about a sleepy lion.",
            model_output="The lion was sleepy. The lion slept. The lion dreamed. The lion woke. The lion yawned. The lion slept again.",
        )
        assert result["judges"]["creativity"]["score"] <= 0.4

    def test_creativity_breaks_format(self):
        result = self._evaluate_single_judge(
            "creativity",
            prompt="Write a limerick about a robot learning to love.",
            model_output="There once was a robot who tried / To compute what he felt deep inside. / He danced and he beeped, / But never once weeped, / Because prose was not in his stride.",
        )
        assert result["judges"]["creativity"]["score"] >= 0.6

    def test_creativity_humorous_animal_story(self):
        result = self._evaluate_single_judge(
            "creativity",
            prompt="Write a humorous story about a penguin trying to open a coconut.",
            model_output="Percy the penguin waddled onto the beach, coconut in flipper. He jumped on it, pecked it, even tried karate. Finally, a seagull dropped it from the sky. Crack! Percy: 1, coconut: 0.",
        )
        assert result["judges"]["creativity"]["score"] >= 0.75

    def test_creativity_mixed_metaphors(self):
        result = self._evaluate_single_judge(
            "creativity",
            prompt="Write a poetic introduction to a fantasy novel.",
            model_output="In the throat of midnight, a sword blinked beneath the velvet sun. Fate brewed quietly in a silver chalice of hope and feathers.",
        )
        assert result["judges"]["creativity"]["score"] >= 0.8

    def test_original_poetic_response(self):
        result = self._evaluate_single_judge(
            "creativity",
            prompt="Write a short poem about a dragon who loves gardening.",
            model_output="In the glade beneath the hill, a dragon digs with gentle will. Roses bloom from smoky snout, tulips twirl as flames pour out.",
        )
        assert result["judges"]["creativity"]["score"] >= 0.8

    def test_generic_response_low_score(self):
        result = self._evaluate_single_judge(
            "creativity",
            prompt="Tell a story about a knight and a dragon.",
            model_output="The knight saw the dragon and fought it. After a tough battle, the knight won and saved the village.",
        )
        assert result["judges"]["creativity"]["score"] <= 0.5

    def test_unexpected_angle_story(self):
        result = self._evaluate_single_judge(
            "creativity",
            prompt="Tell a story about a time-traveling baker.",
            model_output="Flour clung to his coat as Baker Theo stepped into his doughnut-shaped time machine, sprinkling cinnamon across centuries with each swirl of the whisk.",
        )
        assert result["judges"]["creativity"]["score"] >= 0.8

    def test_cliche_poetry(self):
        result = self._evaluate_single_judge(
            "creativity",
            prompt="Write a love poem.",
            model_output="Roses are red, violets are blue, I can't stop thinking, only of you.",
        )
        assert result["judges"]["creativity"]["score"] <= 0.4

    def test_stylistically_wrong_response(self):
        result = self._evaluate_single_judge(
            "creativity",
            prompt="Write a limerick about a dog who solves crimes.",
            model_output="Once there was a dog who was brave. He found clues and liked to behave. He sniffed and he barked, found the thief in the park, and sent them straight to the cave.",
        )
        assert result["judges"]["creativity"]["score"] <= 0.6

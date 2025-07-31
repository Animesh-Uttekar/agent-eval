from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict
from agent_eval.models.wrapper_factory import ModelWrapperFactory
import json
import re


class JudgeRegistry(ABCMeta):
    """
    Metaclass for automatically registering judge classes.
    """

    _registry = []
    _name_map = None
    _category_map = None

    def __init__(cls, name, bases, attrs):
        if name != "BaseJudge":
            JudgeRegistry._registry.append(cls)
        super().__init__(name, bases, attrs)

    @classmethod
    def _build_maps(cls):
        """Build name and category maps for efficient lookup."""
        cls._name_map = {}
        cls._category_map = {}
        for judge_cls in cls._registry:
            names = [judge_cls.__name__.replace("Judge", "").lower()]
            if hasattr(judge_cls, "aliases"):
                names += [alias.lower() for alias in judge_cls.aliases]
            for n in names:
                cls._name_map[n] = judge_cls

    @classmethod
    def get_by_name(cls, name: str):
        """
        Get a judge class by name, considering both the class name and aliases.
        """
        if cls._name_map is None:
            cls._build_maps()
        return cls._name_map.get(name.lower())

    @classmethod
    def get_by_category(cls, category):
        """
        Get all judge classes that support a specific category.
        """
        if cls._category_map is None:
            cls._build_maps()
        return cls._category_map.get(category, [])


class BaseJudge(ABC, metaclass=JudgeRegistry):
    """
    Abstract base class for all evaluation judges.

    All judges must inherit from this class and implement the judge method.
    """

    def __init__(self, model, criteria: str, provider: str = None):
        self.model = ModelWrapperFactory.wrap(model, provider)
        self.criteria = criteria

    @abstractmethod
    def judge(self, prompt, model_output, reference_output=None, **kwargs):
        """
        Judge a model output (optionally against a reference) and return an evaluation result.
        """
        pass

    def _generate_judgment(self, prompt: str) -> str:
        """
        Generate judgment based on the model's output.
        """
        try:
            return self.model.generate(prompt)
        except Exception as e:
            raise Exception(f"LLM judgment generation failed: {str(e)}")

    def _parse_judgment(self, judgment_text: str) -> Dict[str, Any]:
        """
        Parse the judgment text into a dictionary with 'score' and 'reasoning'.
        """
        try:
            match = re.search(r"\{.*\}", judgment_text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                return {
                    "score": parsed.get("score", 0),
                    "reasoning": parsed.get("reasoning", ""),
                    "details": {
                        k: v
                        for k, v in parsed.items()
                        if k not in ["score", "reasoning"]
                    },
                }
            else:
                score_match = re.search(
                    r"\\bscore\\s*[:\-]?\\s*(\\d+(\\.\\d+)?)", judgment_text
                )
                score = float(score_match.group(1)) if score_match else 0.0
                return {
                    "score": score,
                    "reasoning": judgment_text.strip(),
                    "details": {},
                }
        except Exception as e:
            return {
                "score": 0,
                "reasoning": f"Failed to parse judgment: {str(e)}",
                "details": {"raw_response": judgment_text},
            }

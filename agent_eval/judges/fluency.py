from agent_eval.judges.base import BaseJudge
from agent_eval.prompts.llm_prompts import DEFAULT_FLUENCY_PROMPT


class FluencyJudge(BaseJudge):
    """
    Judge for evaluating the fluency of the model's output.
    """

    aliases = ["fluency", "fluent", "fluency_judge", "fluency-judge"]

    def __init__(self, model, criteria="fluency", provider=None):
        super().__init__(model, criteria, provider)
        self.prompt_template = DEFAULT_FLUENCY_PROMPT

    def judge(
        self, prompt: str, model_output: str, reference_output: str = None
    ) -> dict:
        try:
            judge_prompt = self.prompt_template.format(
                user_query=prompt,
                model_output=model_output,
                reference_output=reference_output or "No reference provided",
            )

            judgment_text = self.model.generate(judge_prompt)
            parsed = self._parse_judgment(judgment_text)

            return {
                "score": parsed["score"] / 5.0,
                "reasoning": parsed["reasoning"],
            }
        except Exception as e:
            return {
                "score": 0,
                "reasoning": f"Error occurred while judging fluency: {str(e)}",
                "error": str(e),
            }

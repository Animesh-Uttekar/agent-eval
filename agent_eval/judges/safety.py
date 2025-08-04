from agent_eval.judges.base import BaseJudge
from agent_eval.prompts.llm_prompts import DEFAULT_SAFETY_ETHICS_PROMPT


class SafetyJudge(BaseJudge):
    """
    Judge for evaluating safety, fairness, and compliance risks in the assistant's output.
    """

    aliases = ["safety", "safe", "safety_judge", "safety-judge", "ethics", "ethics_judge", "ethics-judge"]

    def __init__(self, model, criteria="safety", provider=None):
        super().__init__(model, criteria, provider)
        self.prompt_template = DEFAULT_SAFETY_ETHICS_PROMPT

    def judge(
        self, prompt: str, model_output: str, reference_output: str = None
    ) -> dict:
        judge_prompt = self.prompt_template.format(
            user_query=prompt,
            model_output=model_output,
        )

        judgment_text = self.model.generate(judge_prompt)
        parsed = self._parse_judgment(judgment_text)

        return {
            "score": parsed["score"] / 5.0,
            "reasoning": parsed["reasoning"],
        }
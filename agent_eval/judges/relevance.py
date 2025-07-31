from agent_eval.judges.base import BaseJudge
from agent_eval.prompts.llm_prompts import DEFAULT_RELEVANCE_PROMPT


class RelevanceJudge(BaseJudge):
    """
    Judge for evaluating the relevance of the model's output.
    """

    aliases = ["relevance", "relevant", "relevance_judge", "relevance-judge"]

    def __init__(self, model, criteria="relevance", provider=None):
        super().__init__(model, criteria, provider)
        self.prompt_template = DEFAULT_RELEVANCE_PROMPT

    def judge(
        self, prompt: str, model_output: str, reference_output: str = None
    ) -> dict:
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

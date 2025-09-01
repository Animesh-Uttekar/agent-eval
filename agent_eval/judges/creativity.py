from agent_eval.judges.base import BaseJudge
from agent_eval.prompts.llm_prompts import DEFAULT_CREATIVITY_ORIGINALITY_PROMPT


class CreativityJudge(BaseJudge):
    """
    Judge for evaluating the creativity and originality of the model's output.
    """

    aliases = [
        "creativity",
        "creative",
        "creativity_judge",
        "creativity-judge",
        "originality",
        "originality_judge",
        "originality-judge",
    ]

    def __init__(self, model, criteria="creativity", provider=None, enable_cot=False):
        super().__init__(model, criteria, provider, enable_cot)

    def judge(self, prompt: str, model_output: str, reference_output: str = None, **kwargs) -> dict:
        """Judge creativity using specialized prompt with optional CoT enhancement."""
        if self.enable_cot:
            return self._evaluate_with_cot_enhancement(prompt, model_output, reference_output)
        else:
            return self._evaluate_with_specialized_prompt(prompt, model_output, reference_output)
    
    def _get_prompt_template(self):
        """Return the creativity-specific prompt template."""
        return DEFAULT_CREATIVITY_ORIGINALITY_PROMPT
    
    def _get_judge_name(self):
        """Return the judge name for domain-specific fields."""
        return "creativity"
    
    def _get_objective_domain_fields(self, parsed: dict, score: float) -> dict:
        """Return objective creativity-specific metrics only."""
        return {
            "creativity_score": score,  # Objective 0-1 score
            "originality_score": parsed.get("originality_score", score),  # If available from prompt
            "style_adherence_score": parsed.get("style_score", score),  # If available from prompt
            "narrative_quality_score": parsed.get("narrative_score", score)  # If available from prompt
        }

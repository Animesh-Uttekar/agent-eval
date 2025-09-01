from agent_eval.judges.base import BaseJudge
from agent_eval.prompts.llm_prompts import DEFAULT_HELPFULNESS_PROMPT


class HelpfulnessJudge(BaseJudge):
    """
    Judge for evaluating the helpfulness of the model's output.
    """

    aliases = ["helpfulness", "helpful", "helpfulness_judge", "helpfulness-judge"]

    def __init__(self, model, criteria="helpfulness", provider=None, enable_cot=False):
        super().__init__(model, criteria, provider, enable_cot)

    def judge(self, prompt: str, model_output: str, reference_output: str = None, **kwargs) -> dict:
        """Judge helpfulness using specialized prompt with optional CoT enhancement."""
        if self.enable_cot:
            return self._evaluate_with_cot_enhancement(prompt, model_output, reference_output)
        else:
            return self._evaluate_with_specialized_prompt(prompt, model_output, reference_output)
    
    def _get_prompt_template(self):
        """Return the helpfulness-specific prompt template."""
        return DEFAULT_HELPFULNESS_PROMPT
    
    def _get_judge_name(self):
        """Return the judge name for domain-specific fields."""
        return "helpfulness"
    
    def _get_objective_domain_fields(self, parsed: dict, score: float) -> dict:
        """Return objective helpfulness-specific metrics only."""
        return {
            "helpfulness_score": score,  # Objective 0-1 score
            "completeness_score": parsed.get("completeness_score", score),  # If available from prompt
            "relevance_score": parsed.get("relevance_score", score),  # If available from prompt
            "actionable_guidance_score": parsed.get("actionable_score", score)  # If available from prompt
        }

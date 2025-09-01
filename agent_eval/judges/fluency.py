from agent_eval.judges.base import BaseJudge
from agent_eval.prompts.llm_prompts import DEFAULT_FLUENCY_PROMPT


class FluencyJudge(BaseJudge):
    """
    Judge for evaluating the fluency of the model's output.
    """

    aliases = ["fluency", "fluent", "fluency_judge", "fluency-judge"]

    def __init__(self, model, criteria="fluency", provider=None, enable_cot=False):
        super().__init__(model, criteria, provider, enable_cot)

    def judge(self, prompt: str, model_output: str, reference_output: str = None, **kwargs) -> dict:
        """Judge fluency using specialized prompt with optional CoT enhancement."""
        if self.enable_cot:
            return self._evaluate_with_cot_enhancement(prompt, model_output, reference_output)
        else:
            return self._evaluate_with_specialized_prompt(prompt, model_output, reference_output)
    
    def _get_prompt_template(self):
        """Return the fluency-specific prompt template."""
        return DEFAULT_FLUENCY_PROMPT
    
    def _get_judge_name(self):
        """Return the judge name for domain-specific fields."""
        return "fluency"
    
    def _get_objective_domain_fields(self, parsed: dict, score: float) -> dict:
        """Return objective fluency-specific metrics only."""
        return {
            "fluency_score": score,  # Objective 0-1 score
            "grammar_score": parsed.get("grammar_score", score),  # If available from prompt
            "coherence_score": parsed.get("coherence_score", score),  # If available from prompt
            "readability_score": parsed.get("readability_score", score)  # If available from prompt
        }

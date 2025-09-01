from agent_eval.judges.base import BaseJudge
from agent_eval.prompts.llm_prompts import DEFAULT_FACTUALITY_PROMPT


class FactualityJudge(BaseJudge):
    """
    Enhanced factuality judge using Chain-of-Thought reasoning.
    
    Evaluates factual accuracy with systematic bias detection and correction.
    """

    aliases = ["factuality", "factual", "factuality_judge", "factuality-judge"]

    def __init__(self, model, criteria="factuality", provider=None, enable_cot=False):
        super().__init__(model, criteria, provider, enable_cot)

    def judge(self, prompt: str, model_output: str, reference_output: str = None, **kwargs) -> dict:
        """Judge factual accuracy using specialized prompt with optional CoT enhancement."""
        if self.enable_cot:
            return self._evaluate_with_cot_enhancement(prompt, model_output, reference_output)
        else:
            return self._evaluate_with_specialized_prompt(prompt, model_output, reference_output)
    
    def _get_prompt_template(self):
        """Return the factuality-specific prompt template."""
        return DEFAULT_FACTUALITY_PROMPT
    
    def _get_judge_name(self):
        """Return the judge name for domain-specific fields."""
        return "factuality"
    
    def _get_objective_domain_fields(self, parsed: dict, score: float) -> dict:
        """Return objective factuality-specific metrics only."""
        return {
            "factual_accuracy_score": score,  # Objective 0-1 score
            "requires_fact_checking": score < 0.7,  # Objective threshold
            "citation_quality_score": parsed.get("citation_score", score),  # If available from prompt
            "claim_verification_score": parsed.get("verification_score", score)  # If available from prompt
        }

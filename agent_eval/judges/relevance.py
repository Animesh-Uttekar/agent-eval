from agent_eval.judges.base import BaseJudge
from agent_eval.prompts.llm_prompts import DEFAULT_RELEVANCE_PROMPT


class RelevanceJudge(BaseJudge):
    """
    Judge for evaluating the relevance of the model's output.
    """

    aliases = ["relevance", "relevant", "relevance_judge", "relevance-judge"]

    def __init__(self, model, criteria="relevance", provider=None, enable_cot=False):
        super().__init__(model, criteria, provider, enable_cot)

    def judge(self, prompt: str, model_output: str, reference_output: str = None, **kwargs) -> dict:
        """Judge relevance using specialized prompt with optional CoT enhancement."""
        if self.enable_cot:
            return self._evaluate_with_cot_enhancement(prompt, model_output, reference_output)
        else:
            return self._evaluate_with_specialized_prompt(prompt, model_output, reference_output)
    
    def _get_prompt_template(self):
        """Return the relevance-specific prompt template."""
        return DEFAULT_RELEVANCE_PROMPT
    
    def _get_judge_name(self):
        """Return the judge name for domain-specific fields."""
        return "relevance"
    
    def _get_objective_domain_fields(self, parsed: dict, score: float) -> dict:
        """Return objective relevance-specific metrics only."""
        return {
            "relevance_score": score,  # Objective 0-1 score
            "context_alignment_score": parsed.get("context_score", score),  # If available from prompt
            "topic_coverage_score": parsed.get("coverage_score", score),  # If available from prompt
            "query_addressing_score": parsed.get("addressing_score", score)  # If available from prompt
        }

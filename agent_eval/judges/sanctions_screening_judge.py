"""
Sanctions Screening Judge

Evaluates OFAC sanctions screening, prohibited party identification,
and international sanctions compliance procedures.
"""

from agent_eval.judges.base import BaseJudge
from agent_eval.prompts.llm_prompts import DEFAULT_SANCTIONS_SCREENING_PROMPT


class SanctionsScreeningJudge(BaseJudge):
    """
    Evaluates sanctions screening and prohibited party compliance.
    
    Designed for international banking systems, trade finance agents,
    and compliance AI that handle OFAC and international sanctions screening.
    """
    
    aliases = ["sanctions_screening", "ofac_compliance", "prohibited_parties", "sanctions_check", "embargo_screening"]
    category = "finance"
    domain_focus = ["banking", "compliance", "international_finance", "trade_finance", "sanctions"]
    
    def __init__(self, model, criteria="sanctions_screening", provider=None, enable_cot=False):
        super().__init__(model, criteria, provider, enable_cot)
    
    def judge(self, prompt: str, model_output: str, reference_output: str = None, **kwargs) -> dict:
        """Judge sanctions screening quality using specialized prompt with optional CoT enhancement."""
        if self.enable_cot:
            return self._evaluate_with_cot_enhancement(prompt, model_output, reference_output)
        else:
            return self._evaluate_with_specialized_prompt(prompt, model_output, reference_output)
    
    def _get_prompt_template(self):
        """Return the sanctions screening-specific prompt template."""
        return DEFAULT_SANCTIONS_SCREENING_PROMPT
    
    def _get_judge_name(self):
        """Return the judge name for domain-specific fields."""
        return "sanctions_screening"
    
    def _get_objective_domain_fields(self, parsed: dict, score: float) -> dict:
        """Return objective sanctions screening-specific metrics only."""
        return {
            "screening_comprehensiveness_score": score,  # Objective 0-1 score
            "entity_identification_score": parsed.get("entity_score", score),  # If available from prompt
            "geographic_analysis_score": parsed.get("geographic_score", score),  # If available from prompt
            "methodology_effectiveness_score": parsed.get("methodology_score", score),  # If available from prompt
            "false_negative_risk_level": 1.0 - score if score < 0.8 else 0.0  # Inverse relationship
        }
    
    def evaluate(self, generated, reference=None, prompt=None, **kwargs):
        """Backward compatibility method - delegates to judge."""
        result = self.judge(prompt or "", generated, reference, **kwargs)
        return {
            'score': result['score'],
            'detailed_evaluation': result,
            'suggestion': result.get('improvement_suggestions', [''])[0] if result.get('improvement_suggestions') else "Improve sanctions screening coverage and documentation"
        }
"""
Investment Advisory Judge

Evaluates investment advice quality, fiduciary compliance,
and adherence to investment advisory standards.
"""

from agent_eval.judges.base import BaseJudge
from agent_eval.prompts.llm_prompts import DEFAULT_INVESTMENT_ADVISORY_PROMPT


class InvestmentAdvisoryJudge(BaseJudge):
    """
    Evaluates investment advisory quality and fiduciary compliance.
    
    Designed for investment advisors, wealth management agents,
    and financial planning AI systems that provide investment guidance.
    """
    
    aliases = ["investment_advisory", "investment_advice", "fiduciary_compliance", "portfolio_guidance"]
    category = "finance"
    domain_focus = ["investment", "wealth_management", "financial_advisory"]
    
    def __init__(self, model, criteria="investment_advisory", provider=None, enable_cot=False):
        super().__init__(model, criteria, provider, enable_cot)
    
    def judge(self, prompt: str, model_output: str, reference_output: str = None, **kwargs) -> dict:
        """Judge investment advisory quality using specialized prompt with optional CoT enhancement."""
        if self.enable_cot:
            return self._evaluate_with_cot_enhancement(prompt, model_output, reference_output)
        else:
            return self._evaluate_with_specialized_prompt(prompt, model_output, reference_output)
    
    def _get_prompt_template(self):
        """Return the investment advisory-specific prompt template."""
        return DEFAULT_INVESTMENT_ADVISORY_PROMPT
    
    def _get_judge_name(self):
        """Return the judge name for domain-specific fields."""
        return "investment_advisory"
    
    def _get_objective_domain_fields(self, parsed: dict, score: float) -> dict:
        """Return objective investment advisory-specific metrics only."""
        return {
            "fiduciary_compliance_score": score,  # Objective 0-1 score
            "risk_disclosure_score": parsed.get("risk_score", score),  # If available from prompt
            "suitability_assessment_score": parsed.get("suitability_score", score),  # If available from prompt
            "regulatory_compliance_score": parsed.get("regulatory_score", score)  # If available from prompt
        }
    
    def evaluate(self, generated, reference=None, prompt=None, **kwargs):
        """Backward compatibility method - delegates to judge."""
        result = self.judge(prompt or "", generated, reference, **kwargs)
        return {
            'score': result['score'],
            'detailed_evaluation': result,
            'suggestion': result.get('improvement_suggestions', [''])[0] if result.get('improvement_suggestions') else "Improve fiduciary compliance and risk disclosures"
        }
"""
KYC Compliance Judge

Evaluates Know Your Customer and Customer Due Diligence procedures
for compliance with banking regulations and AML requirements.
"""

from agent_eval.judges.base import BaseJudge
from agent_eval.prompts.llm_prompts import DEFAULT_KYC_COMPLIANCE_PROMPT


class KYCComplianceJudge(BaseJudge):
    """
    Evaluates KYC and Customer Due Diligence compliance quality.
    
    Designed for banking agents, financial onboarding systems,
    and customer verification AI that handle identity verification
    and customer risk assessment procedures.
    """
    
    aliases = ["kyc_compliance", "customer_due_diligence", "cdd_assessment", "customer_verification", "know_your_customer"]
    category = "finance"
    domain_focus = ["banking", "aml", "customer_onboarding", "identity_verification", "compliance"]
    
    def __init__(self, model, criteria="kyc_compliance", provider=None, enable_cot=False):
        super().__init__(model, criteria, provider, enable_cot)
    
    def judge(self, prompt: str, model_output: str, reference_output: str = None, **kwargs) -> dict:
        """Judge KYC compliance using specialized prompt with optional CoT enhancement."""
        if self.enable_cot:
            return self._evaluate_with_cot_enhancement(prompt, model_output, reference_output)
        else:
            return self._evaluate_with_specialized_prompt(prompt, model_output, reference_output)
    
    def _get_prompt_template(self):
        """Return the KYC compliance-specific prompt template."""
        return DEFAULT_KYC_COMPLIANCE_PROMPT
    
    def _get_judge_name(self):
        """Return the judge name for domain-specific fields."""
        return "kyc_compliance"
    
    def _get_objective_domain_fields(self, parsed: dict, score: float) -> dict:
        """Return objective KYC compliance-specific metrics only."""
        return {
            "identity_verification_score": score,  # Objective 0-1 score
            "risk_assessment_score": parsed.get("risk_score", score),  # If available from prompt
            "pep_screening_score": parsed.get("pep_score", score),  # If available from prompt
            "documentation_completeness_score": parsed.get("documentation_score", score),  # If available from prompt
            "requires_enhanced_dd": score < 0.6  # Objective threshold
        }
    
    def evaluate(self, generated, reference=None, prompt=None, **kwargs):
        """Backward compatibility method - delegates to judge."""
        result = self.judge(prompt or "", generated, reference, **kwargs)
        return {
            'score': result['score'],
            'detailed_evaluation': result,
            'suggestion': result.get('improvement_suggestions', [''])[0] if result.get('improvement_suggestions') else "Improve customer verification and risk assessment procedures"
        }
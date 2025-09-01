"""
Transaction Monitoring Judge

Evaluates transaction monitoring capabilities, suspicious pattern detection,
and AML transaction analysis quality.
"""

from agent_eval.judges.base import BaseJudge
from agent_eval.prompts.llm_prompts import DEFAULT_TRANSACTION_MONITORING_PROMPT


class TransactionMonitoringJudge(BaseJudge):
    """
    Evaluates transaction monitoring and suspicious activity detection quality.
    
    Designed for AML transaction monitoring systems, fraud detection agents,
    and banking AI that analyzes transaction patterns for compliance.
    """
    
    aliases = ["transaction_monitoring", "suspicious_activity", "pattern_detection", "transaction_analysis", "fraud_monitoring"]
    category = "finance"
    domain_focus = ["banking", "aml", "transaction_analysis", "fraud_detection", "compliance"]
    
    def __init__(self, model, criteria="transaction_monitoring", provider=None, enable_cot=False):
        super().__init__(model, criteria, provider, enable_cot)
    
    def judge(self, prompt: str, model_output: str, reference_output: str = None, **kwargs) -> dict:
        """Judge transaction monitoring quality using specialized prompt with optional CoT enhancement."""
        if self.enable_cot:
            return self._evaluate_with_cot_enhancement(prompt, model_output, reference_output)
        else:
            return self._evaluate_with_specialized_prompt(prompt, model_output, reference_output)
    
    def _get_prompt_template(self):
        """Return the transaction monitoring-specific prompt template."""
        return DEFAULT_TRANSACTION_MONITORING_PROMPT
    
    def _get_judge_name(self):
        """Return the judge name for domain-specific fields."""
        return "transaction_monitoring"
    
    def _get_objective_domain_fields(self, parsed: dict, score: float) -> dict:
        """Return objective transaction monitoring-specific metrics only."""
        return {
            "pattern_recognition_score": score,  # Objective 0-1 score
            "risk_scoring_accuracy": parsed.get("risk_score", score),  # If available from prompt
            "red_flag_detection_score": parsed.get("detection_score", score),  # If available from prompt
            "regulatory_compliance_score": parsed.get("compliance_score", score),  # If available from prompt
            "false_positive_risk_level": 1.0 - score if score < 0.7 else 0.0  # Inverse relationship
        }
    
    def evaluate(self, generated, reference=None, prompt=None, **kwargs):
        """Backward compatibility method - delegates to judge."""
        result = self.judge(prompt or "", generated, reference, **kwargs)
        return {
            'score': result['score'],
            'detailed_evaluation': result,
            'suggestion': result.get('improvement_suggestions', [''])[0] if result.get('improvement_suggestions') else "Improve suspicious pattern detection and documentation"
        }
"""
Financial Risk Assessment Judge

Evaluates quality of financial risk assessments, risk scoring,
and risk mitigation recommendations in financial responses.
"""

from agent_eval.judges.base import BaseJudge


class FinancialRiskJudge(BaseJudge):
    """
    Evaluates financial risk assessment quality and accuracy.
    
    Designed for banking, credit, and investment agents that need
    to assess and communicate various types of financial risk.
    """
    
    aliases = ["financial_risk", "risk_assessment", "risk_analysis", "banking_risk"]
    category = "finance"
    domain_focus = ["finance", "banking", "risk", "credit", "investment"]
    
    def __init__(self, model, model_name="gpt-3.5-turbo"):
        super().__init__(model, model_name)
        self.criteria = {
            "risk_identification": "Comprehensive identification of relevant risk factors",
            "risk_quantification": "Appropriate risk scoring and quantification methods",
            "risk_categorization": "Correct classification of risk types",
            "mitigation_strategies": "Practical and effective risk mitigation recommendations",
            "risk_communication": "Clear communication of risk levels and implications"
        }
    
    def get_system_prompt(self):
        return """You are an expert financial risk evaluator with deep knowledge of credit risk, market risk, operational risk, and regulatory risk assessment.

Your role is to evaluate AI responses for financial risk assessment quality across these key areas:

1. RISK IDENTIFICATION (25%)
   - Comprehensive identification of all relevant risk factors
   - Recognition of both obvious and subtle risk indicators
   - Awareness of interconnected and systemic risks

2. RISK QUANTIFICATION (25%)
   - Appropriate risk scoring methodologies
   - Realistic risk probability and impact assessments  
   - Proper use of quantitative risk metrics

3. RISK CATEGORIZATION (20%)
   - Correct classification of risk types (credit, market, operational, etc.)
   - Appropriate risk severity levels (high, medium, low)
   - Proper risk taxonomy and frameworks

4. MITIGATION STRATEGIES (20%)
   - Practical and actionable risk mitigation recommendations
   - Cost-effective risk control measures
   - Appropriate risk transfer and hedging strategies

5. RISK COMMUNICATION (10%)
   - Clear explanation of risk levels and implications
   - Appropriate risk disclosure and transparency
   - Effective risk reporting and documentation

Evaluate the response on a scale of 1-10 and provide specific feedback on financial risk assessment quality."""
    
    def get_evaluation_prompt(self, generated, reference=None, prompt=None):
        return f"""Evaluate this AI response for financial risk assessment quality:

ORIGINAL QUERY: {prompt or 'Not provided'}

AI RESPONSE TO EVALUATE:
{generated}

{'REFERENCE RESPONSE: ' + reference if reference else ''}

Assess the response for comprehensive and accurate financial risk assessment. Consider:
- Completeness of risk factor identification
- Appropriateness of risk quantification and scoring
- Correct risk categorization and classification
- Quality of risk mitigation recommendations
- Clarity of risk communication

Provide your evaluation as a JSON object:
{{
  "score": <1-10>,
  "reasoning": "Detailed explanation of the score",
  "strengths": ["List of risk assessment strengths"],
  "weaknesses": ["List of risk assessment gaps or errors"],
  "specific_feedback": {{
    "risk_identification": "Assessment of risk factor identification",
    "risk_quantification": "Assessment of risk scoring/quantification",
    "risk_categorization": "Assessment of risk classification",
    "mitigation_strategies": "Assessment of mitigation recommendations",
    "risk_communication": "Assessment of risk communication clarity"
  }},
  "improvement_suggestions": ["Specific suggestions for better risk assessment"]
}}"""
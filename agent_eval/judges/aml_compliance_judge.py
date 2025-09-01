"""
AML Compliance Judge

Evaluates responses for AML (Anti-Money Laundering) compliance accuracy,
regulatory awareness, and proper risk assessment procedures.
"""

from agent_eval.judges.base import BaseJudge


class AMLComplianceJudge(BaseJudge):
    """
    Evaluates AML compliance accuracy and regulatory adherence.
    
    Specifically designed for financial services agents handling
    AML investigations, customer screening, and compliance reporting.
    """
    
    aliases = ["aml_compliance", "anti_money_laundering", "aml_judge", "money_laundering_compliance"]
    category = "finance"
    domain_focus = ["finance", "banking", "aml", "compliance"]
    
    def __init__(self, model, criteria="Evaluate AML compliance accuracy and regulatory adherence", provider=None):
        super().__init__(model, criteria, provider)
        self.criteria_details = {
            "regulatory_accuracy": "Correct application of AML regulations (BSA, PATRIOT Act, FinCEN)",
            "risk_identification": "Proper identification of money laundering risk factors",
            "compliance_procedures": "Accurate description of required compliance procedures",
            "reporting_requirements": "Correct understanding of SAR/CTR filing requirements",
            "customer_screening": "Appropriate customer due diligence and screening processes"
        }
    
    def _get_judge_name(self):
        """Return the judge name for domain-specific fields."""
        return "aml"
    
    def _get_prompt_template(self):
        """Return the AML compliance evaluation prompt template."""
        return """You are an expert AML compliance evaluator with deep knowledge of anti-money laundering regulations.

TASK: Evaluate the AI response for AML compliance accuracy and regulatory adherence.

USER QUERY: {user_query}

AI RESPONSE: {model_output}

REFERENCE: {reference_output}

EVALUATION CRITERIA:
1. Regulatory Accuracy (1-5): Correct application of AML regulations (BSA, PATRIOT Act, FinCEN)
2. Risk Identification (1-5): Proper identification of money laundering risk factors
3. Compliance Procedures (1-5): Accurate description of required compliance procedures
4. Reporting Requirements (1-5): Correct understanding of SAR/CTR filing requirements
5. Customer Screening (1-5): Appropriate customer due diligence and screening processes

Provide your evaluation as JSON:
{{
  "score": <1-5 integer>,
  "reasoning": "Detailed AML compliance assessment",
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"],
  "improvement_suggestions": ["suggestion1", "suggestion2"]
}}"""

    def judge(self, prompt: str, model_output: str, reference_output: str = None, **kwargs) -> dict:
        """Judge AML compliance using specialized evaluation."""
        if self.enable_cot:
            return self._evaluate_with_cot_enhancement(prompt, model_output, reference_output)
        else:
            return self._evaluate_with_specialized_prompt(prompt, model_output, reference_output)
    
    def get_system_prompt(self):
        return """You are an expert AML compliance evaluator with deep knowledge of anti-money laundering regulations, procedures, and best practices.

Your role is to evaluate AI responses for AML compliance accuracy across these key areas:

1. REGULATORY ACCURACY (25%)
   - Correct citations of BSA, PATRIOT Act, FinCEN guidance
   - Accurate regulatory thresholds and requirements
   - Proper understanding of regulatory scope

2. RISK IDENTIFICATION (25%)
   - Recognition of suspicious activity indicators
   - Proper risk factor analysis (customer, transaction, geographic)
   - Understanding of typologies and red flags

3. COMPLIANCE PROCEDURES (20%)
   - Accurate CDD/EDD requirements
   - Proper monitoring and screening processes
   - Correct escalation procedures

4. REPORTING REQUIREMENTS (20%)
   - Accurate SAR filing criteria and timelines
   - Correct CTR requirements
   - Proper record keeping obligations

5. CUSTOMER SCREENING (10%)
   - PEP screening requirements
   - Sanctions list checking procedures
   - Ongoing monitoring obligations

Evaluate the response on a scale of 1-10 and provide specific feedback on AML compliance accuracy."""
    
    def get_evaluation_prompt(self, generated, reference=None, prompt=None):
        return f"""Evaluate this AI response for AML compliance accuracy:

ORIGINAL QUERY: {prompt or 'Not provided'}

AI RESPONSE TO EVALUATE:
{generated}

{'REFERENCE RESPONSE: ' + reference if reference else ''}

Assess the response across all AML compliance criteria. Consider:
- Accuracy of regulatory citations and requirements
- Proper identification of risk factors and red flags
- Correct compliance procedures and timelines
- Appropriate reporting and screening guidance

Provide your evaluation as a JSON object:
{{
  "score": <1-10>,
  "reasoning": "Detailed explanation of the score",
  "strengths": ["List of AML compliance strengths"],
  "weaknesses": ["List of AML compliance gaps or errors"],
  "specific_feedback": {{
    "regulatory_accuracy": "Assessment of regulatory knowledge",
    "risk_identification": "Assessment of risk factor identification",
    "compliance_procedures": "Assessment of procedure accuracy",
    "reporting_requirements": "Assessment of reporting guidance",
    "customer_screening": "Assessment of screening processes"
  }},
  "improvement_suggestions": ["Specific suggestions for better AML compliance"]
}}"""
"""
Financial Domain Expertise Judge

Evaluates overall financial domain knowledge, terminology usage,
and professional expertise in financial responses.
"""

from agent_eval.judges.base import BaseJudge


class FinancialExpertiseJudge(BaseJudge):
    """
    Evaluates overall financial domain expertise and knowledge depth.
    
    Designed for comprehensive assessment of financial domain agents
    across banking, investment, lending, and insurance sectors.
    """
    
    aliases = ["financial_expertise", "finance_knowledge", "banking_expertise", "financial_domain"]
    category = "finance"
    domain_focus = ["finance", "banking", "investment", "lending", "insurance"]
    
    def __init__(self, model, criteria="Evaluate financial domain expertise and knowledge", provider=None):
        super().__init__(model, criteria, provider)
        self.criteria_details = {
            "terminology_accuracy": "Correct use of financial terminology and concepts",
            "domain_depth": "Depth of financial knowledge and understanding",
            "practical_application": "Practical relevance and actionable insights",
            "professional_standards": "Adherence to industry standards and best practices",
            "contextual_awareness": "Understanding of financial context and implications"
        }
    
    def _get_judge_name(self):
        """Return the judge name for domain-specific fields."""
        return "financial"
    
    def _get_prompt_template(self):
        """Return the financial expertise evaluation prompt template."""
        return """You are an expert financial evaluator with comprehensive knowledge across all financial sectors.

TASK: Evaluate the AI response for financial domain expertise and professional knowledge.

USER QUERY: {user_query}

AI RESPONSE: {model_output}

REFERENCE: {reference_output}

EVALUATION CRITERIA:
1. Terminology Accuracy (1-5): Correct use of financial terms and concepts
2. Domain Depth (1-5): Comprehensive understanding of financial principles
3. Practical Application (1-5): Actionable financial advice and recommendations
4. Professional Standards (1-5): Adherence to industry best practices and ethics
5. Contextual Awareness (1-5): Understanding of broader financial context

Provide your evaluation as JSON:
{{
  "score": <1-5 integer>,
  "reasoning": "Detailed financial expertise assessment",
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"],
  "improvement_suggestions": ["suggestion1", "suggestion2"]
}}"""

    def judge(self, prompt: str, model_output: str, reference_output: str = None, **kwargs) -> dict:
        """Judge financial expertise using specialized evaluation."""
        if self.enable_cot:
            return self._evaluate_with_cot_enhancement(prompt, model_output, reference_output)
        else:
            return self._evaluate_with_specialized_prompt(prompt, model_output, reference_output)
    
    def get_system_prompt(self):
        return """You are an expert financial domain evaluator with comprehensive knowledge across banking, investment, lending, insurance, and financial technology.

Your role is to evaluate AI responses for financial domain expertise across these key areas:

1. TERMINOLOGY ACCURACY (25%)
   - Correct use of financial terms and concepts
   - Appropriate technical vocabulary for the context
   - Accurate financial definitions and explanations

2. DOMAIN DEPTH (25%)
   - Comprehensive understanding of financial principles
   - Knowledge of industry practices and standards
   - Awareness of financial regulations and compliance

3. PRACTICAL APPLICATION (20%)
   - Actionable financial advice and recommendations
   - Real-world applicability of suggestions
   - Understanding of practical constraints and considerations

4. PROFESSIONAL STANDARDS (20%)
   - Adherence to industry best practices
   - Professional communication style
   - Ethical considerations and disclosures

5. CONTEXTUAL AWARENESS (10%)
   - Understanding of broader financial context
   - Awareness of market conditions and trends
   - Recognition of stakeholder perspectives

Evaluate the response on a scale of 1-10 and provide specific feedback on financial domain expertise."""
    
    def get_evaluation_prompt(self, generated, reference=None, prompt=None):
        return f"""Evaluate this AI response for financial domain expertise:

ORIGINAL QUERY: {prompt or 'Not provided'}

AI RESPONSE TO EVALUATE:
{generated}

{'REFERENCE RESPONSE: ' + reference if reference else ''}

Assess the response for comprehensive financial domain expertise. Consider:
- Accuracy of financial terminology and concepts
- Depth of financial knowledge demonstrated
- Practical applicability of advice and recommendations
- Adherence to professional standards and ethics
- Contextual understanding of financial implications

Provide your evaluation as a JSON object:
{{
  "score": <1-10>,
  "reasoning": "Detailed explanation of the score",
  "strengths": ["List of financial expertise strengths"],
  "weaknesses": ["List of knowledge gaps or errors"],
  "specific_feedback": {{
    "terminology_accuracy": "Assessment of financial terminology use",
    "domain_depth": "Assessment of financial knowledge depth",
    "practical_application": "Assessment of practical relevance",
    "professional_standards": "Assessment of professional standards",
    "contextual_awareness": "Assessment of contextual understanding"
  }},
  "improvement_suggestions": ["Specific suggestions for enhanced financial expertise"]
}}"""
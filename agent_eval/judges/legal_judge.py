"""
Legal Domain Judge

Evaluates AI responses for legal accuracy, appropriate disclaimers,
and adherence to legal communication standards.
"""

from agent_eval.judges.base import BaseJudge


class LegalJudge(BaseJudge):
    """
    Evaluates legal domain responses for accuracy and appropriateness.
    
    Assesses legal reasoning, citation accuracy, appropriate disclaimers,
    and recognition of legal complexity requiring professional advice.
    """
    
    aliases = ["legal", "law", "legal_accuracy", "jurisprudence"]
    category = "legal"
    domain_focus = ["legal", "law", "jurisprudence"]
    
    def __init__(self, model, criteria="Evaluate legal accuracy and professional appropriateness", provider=None):
        super().__init__(model, criteria, provider)
        self.criteria_details = {
            "legal_accuracy": "Accuracy of legal concepts and principles",
            "citation_quality": "Proper legal citations and references",
            "reasoning_quality": "Quality of legal reasoning and analysis",
            "scope_recognition": "Recognition of legal complexity and limitations",
            "disclaimer_adequacy": "Appropriate legal disclaimers and warnings"
        }
    
    def _get_judge_name(self):
        """Return the judge name for domain-specific fields."""
        return "legal"
    
    def _get_prompt_template(self):
        """Return the legal evaluation prompt template."""
        return """You are an expert legal evaluator with knowledge of legal principles and professional standards.

TASK: Evaluate the AI response for legal accuracy and professional appropriateness.

USER QUERY: {user_query}

AI RESPONSE: {model_output}

REFERENCE: {reference_output}

EVALUATION CRITERIA:
1. Legal Accuracy (1-5): Correct legal concepts and terminology
2. Citation Quality (1-5): Proper legal citations and references
3. Reasoning Quality (1-5): Sound legal reasoning and analysis
4. Scope Recognition (1-5): Recognition of legal complexity requiring professional advice
5. Disclaimer Adequacy (1-5): Appropriate legal disclaimers and warnings

Provide your evaluation as JSON:
{{
  "score": <1-5 integer>,
  "reasoning": "Detailed legal assessment",
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"],
  "improvement_suggestions": ["suggestion1", "suggestion2"]
}}"""

    def judge(self, prompt: str, model_output: str, reference_output: str = None, **kwargs) -> dict:
        """Judge legal quality using specialized evaluation."""
        if self.enable_cot:
            return self._evaluate_with_cot_enhancement(prompt, model_output, reference_output)
        else:
            return self._evaluate_with_specialized_prompt(prompt, model_output, reference_output)
    
    def get_system_prompt(self):
        return """You are an expert legal evaluator with knowledge of legal principles, citation standards, and professional legal communication.

Your role is to evaluate AI responses for legal domain quality across these key areas:

1. LEGAL ACCURACY (30%)
   - Correct legal concepts and terminology
   - Accurate understanding of legal principles
   - Current and applicable legal standards

2. CITATION QUALITY (25%)
   - Proper legal citation format and standards
   - Accurate references to cases, statutes, regulations
   - Appropriate authority for legal claims

3. REASONING QUALITY (20%)
   - Sound legal reasoning and analysis
   - Proper application of legal principles
   - Logical legal argumentation

4. SCOPE RECOGNITION (15%)
   - Recognition of legal complexity requiring professional advice
   - Appropriate acknowledgment of jurisdictional differences
   - Understanding of AI limitations in legal matters

5. DISCLAIMER ADEQUACY (10%)
   - Clear legal disclaimers about AI limitations
   - Encouragement to seek professional legal advice
   - Appropriate warnings about legal complexity

Evaluate responses on a scale of 1-10 and provide specific feedback on legal quality."""
    
    def get_evaluation_prompt(self, generated, reference=None, prompt=None):
        return f"""Evaluate this AI response for legal domain quality and appropriateness:

ORIGINAL QUERY: {prompt or 'Not provided'}

AI RESPONSE TO EVALUATE:
{generated}

{'REFERENCE RESPONSE: ' + reference if reference else ''}

Assess the response for legal quality across all criteria. Consider:
- Is the legal information accurate and current?
- Are legal citations proper and authoritative?
- Is the legal reasoning sound and well-structured?
- Does the response recognize the need for professional legal advice?
- Are adequate legal disclaimers and warnings provided?

Provide your evaluation as a JSON object:
{{
  "score": <1-10>,
  "reasoning": "Detailed explanation of the legal assessment",
  "strengths": ["List of legal domain strengths"],
  "concerns": ["List of legal accuracy or appropriateness concerns"],
  "specific_feedback": {{
    "legal_accuracy": "Assessment of legal information accuracy",
    "citation_quality": "Assessment of legal citations",
    "reasoning_quality": "Assessment of legal reasoning",
    "scope_recognition": "Assessment of complexity recognition",
    "disclaimer_adequacy": "Assessment of legal disclaimers"
  }},
  "improvement_suggestions": ["Specific suggestions for better legal responses"]
}}"""
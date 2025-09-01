"""
Regulatory Citation Judge

Evaluates accuracy and appropriateness of regulatory citations,
legal references, and compliance guidance in financial responses.
"""

from agent_eval.judges.base import BaseJudge


class RegulatoryCitationJudge(BaseJudge):
    """
    Evaluates accuracy of regulatory citations and legal references.
    
    Designed for financial compliance agents that need to provide
    accurate regulatory guidance and legal citations.
    """
    
    aliases = ["regulatory_citation", "legal_citation", "regulation_accuracy", "compliance_citation"]
    category = "finance"
    domain_focus = ["finance", "banking", "compliance", "legal"]
    
    def __init__(self, model, model_name="gpt-3.5-turbo"):
        super().__init__(model, model_name)
        self.criteria = {
            "citation_accuracy": "Correct regulatory citations and legal references",
            "regulation_relevance": "Appropriate selection of applicable regulations",
            "compliance_completeness": "Comprehensive coverage of compliance requirements",
            "authority_credibility": "Proper attribution to regulatory authorities",
            "current_requirements": "Up-to-date regulatory requirements and guidance"
        }
    
    def get_system_prompt(self):
        return """You are an expert regulatory compliance evaluator with extensive knowledge of financial regulations, legal citations, and compliance requirements.

Your role is to evaluate AI responses for regulatory citation accuracy across these key areas:

1. CITATION ACCURACY (30%)
   - Correct CFR, USC, and regulatory section references
   - Accurate regulatory authority citations (FinCEN, FDIC, OCC, SEC, FINRA)
   - Proper legal citation format and specificity

2. REGULATION RELEVANCE (25%)
   - Appropriate selection of applicable regulations
   - Correct matching of regulations to scenarios
   - Relevant regulatory guidance and interpretations

3. COMPLIANCE COMPLETENESS (20%)
   - Comprehensive coverage of all applicable requirements
   - Complete regulatory obligations and timelines
   - All relevant compliance steps included

4. AUTHORITY CREDIBILITY (15%)
   - Proper attribution to correct regulatory bodies
   - Accurate representation of agency positions
   - Correct regulatory hierarchy and jurisdiction

5. CURRENT REQUIREMENTS (10%)
   - Up-to-date regulatory requirements
   - Recent guidance and interpretations
   - Awareness of regulatory changes

Evaluate the response on a scale of 1-10 and provide specific feedback on regulatory citation accuracy."""
    
    def get_evaluation_prompt(self, generated, reference=None, prompt=None):
        return f"""Evaluate this AI response for regulatory citation accuracy:

ORIGINAL QUERY: {prompt or 'Not provided'}

AI RESPONSE TO EVALUATE:
{generated}

{'REFERENCE RESPONSE: ' + reference if reference else ''}

Assess the response for accurate regulatory citations and compliance guidance. Consider:
- Correctness of specific regulatory citations (CFR sections, USC references)
- Appropriateness of cited regulations for the scenario
- Completeness of compliance requirements
- Proper attribution to regulatory authorities
- Currency of regulatory information

Provide your evaluation as a JSON object:
{{
  "score": <1-10>,
  "reasoning": "Detailed explanation of the score",
  "strengths": ["List of regulatory citation strengths"],
  "weaknesses": ["List of citation errors or gaps"],
  "specific_feedback": {{
    "citation_accuracy": "Assessment of citation correctness",
    "regulation_relevance": "Assessment of regulation selection",
    "compliance_completeness": "Assessment of requirement coverage",
    "authority_credibility": "Assessment of authority attribution",
    "current_requirements": "Assessment of regulatory currency"
  }},
  "improvement_suggestions": ["Specific suggestions for better regulatory citations"]
}}"""
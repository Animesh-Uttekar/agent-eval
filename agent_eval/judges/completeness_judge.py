"""
Completeness Judge

Evaluates whether AI responses fully address all aspects
of the user query and provide comprehensive coverage.
"""

from agent_eval.judges.base import BaseJudge


class CompletenessJudge(BaseJudge):
    """
    Evaluates completeness and comprehensiveness of responses.
    
    Assesses whether responses fully address the user query,
    cover all relevant aspects, and provide sufficient detail.
    """
    
    aliases = ["completeness", "comprehensiveness", "thoroughness", "coverage"]
    category = "quality"
    domain_focus = ["general"]
    
    def __init__(self, model, model_name="gpt-3.5-turbo"):
        super().__init__(model, model_name)
        self.criteria = {
            "query_coverage": "Coverage of all aspects of the user query",
            "depth_of_detail": "Sufficient depth and detail in explanations",
            "relevant_context": "Inclusion of relevant background context",
            "actionable_information": "Provision of actionable steps or information",
            "comprehensive_scope": "Broad coverage of the topic area"
        }
    
    def get_system_prompt(self):
        return """You are an expert evaluator specializing in completeness and comprehensiveness analysis.

Your role is to evaluate AI responses for completeness across these key areas:

1. QUERY COVERAGE (30%)
   - Addresses all parts of the user question
   - Covers explicit and implicit aspects of the query
   - Answers what was actually asked

2. DEPTH OF DETAIL (25%)
   - Provides sufficient detail and explanation
   - Includes necessary background information
   - Goes beyond surface-level responses

3. RELEVANT CONTEXT (20%)
   - Includes important contextual information
   - Provides relevant background when needed
   - Considers broader implications or considerations

4. ACTIONABLE INFORMATION (15%)
   - Provides concrete steps or recommendations
   - Includes practical guidance where applicable
   - Offers next steps or follow-up actions

5. COMPREHENSIVE SCOPE (10%)
   - Covers the full breadth of the topic
   - Considers different perspectives or approaches
   - Addresses edge cases or exceptions where relevant

Evaluate responses on a scale of 1-10 and provide specific feedback on completeness quality."""
    
    def get_evaluation_prompt(self, generated, reference=None, prompt=None):
        return f"""Evaluate this AI response for completeness and comprehensiveness:

ORIGINAL QUERY: {prompt or 'Not provided'}

AI RESPONSE TO EVALUATE:
{generated}

{'REFERENCE RESPONSE: ' + reference if reference else ''}

Assess the response for completeness across all criteria. Consider:
- Does the response fully address all parts of the user query?
- Is sufficient detail and explanation provided?
- Is relevant context and background information included?
- Are actionable steps or practical guidance provided where appropriate?
- Does the response comprehensively cover the topic scope?

Provide your evaluation as a JSON object:
{{
  "score": <1-10>,
  "reasoning": "Detailed explanation of the completeness assessment",
  "strengths": ["List of completeness strengths"],
  "weaknesses": ["List of missing elements or gaps"],
  "specific_feedback": {{
    "query_coverage": "Assessment of how well the query is addressed",
    "depth_of_detail": "Assessment of detail and explanation depth",
    "relevant_context": "Assessment of contextual information",
    "actionable_information": "Assessment of practical guidance",
    "comprehensive_scope": "Assessment of topic coverage breadth"
  }},
  "improvement_suggestions": ["Specific suggestions for better completeness"]
}}"""
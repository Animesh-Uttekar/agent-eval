"""
Technical Accuracy Judge

Evaluates AI responses for technical correctness, implementation
details, and engineering best practices across technology domains.
"""

from agent_eval.judges.base import BaseJudge


class TechnicalAccuracyJudge(BaseJudge):
    """
    Evaluates technical accuracy and engineering quality.
    
    Assesses correctness of technical information, code examples,
    architectural decisions, and best practices.
    """
    
    aliases = ["technical_accuracy", "engineering", "technical", "code_quality"]
    category = "technical"
    domain_focus = ["technology", "engineering", "software", "programming"]
    
    def __init__(self, model, criteria="Evaluate technical accuracy and engineering quality", provider=None):
        super().__init__(model, criteria, provider)
        self.criteria_details = {
            "factual_correctness": "Accuracy of technical facts and concepts",
            "implementation_quality": "Quality of code examples and implementations",
            "best_practices": "Adherence to engineering best practices",
            "security_awareness": "Consideration of security implications",
            "performance_considerations": "Awareness of performance and scalability"
        }
    
    def _get_judge_name(self):
        """Return the judge name for domain-specific fields."""
        return "technical"
    
    def _get_prompt_template(self):
        """Return the technical accuracy evaluation prompt template."""
        return """You are an expert technical evaluator with deep knowledge of software engineering.

TASK: Evaluate the AI response for technical accuracy and engineering quality.

USER QUERY: {user_query}

AI RESPONSE: {model_output}

REFERENCE: {reference_output}

EVALUATION CRITERIA:
1. Factual Correctness (1-5): Accuracy of technical facts and concepts
2. Implementation Quality (1-5): Quality of code examples and implementations
3. Best Practices (1-5): Adherence to engineering best practices
4. Security Awareness (1-5): Consideration of security implications
5. Performance Considerations (1-5): Awareness of performance and scalability

Provide your evaluation as JSON:
{{
  "score": <1-5 integer>,
  "reasoning": "Detailed technical assessment",
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"],
  "improvement_suggestions": ["suggestion1", "suggestion2"]
}}"""

    def judge(self, prompt: str, model_output: str, reference_output: str = None, **kwargs) -> dict:
        """Judge technical accuracy using specialized evaluation."""
        if self.enable_cot:
            return self._evaluate_with_cot_enhancement(prompt, model_output, reference_output)
        else:
            return self._evaluate_with_specialized_prompt(prompt, model_output, reference_output)
    
    def get_system_prompt(self):
        return """You are an expert technical evaluator with deep knowledge of software engineering, system design, and technology best practices.

Your role is to evaluate AI responses for technical accuracy across these key areas:

1. FACTUAL CORRECTNESS (30%)
   - Accurate technical concepts and terminology
   - Correct understanding of technologies and frameworks
   - Up-to-date knowledge of technical standards

2. IMPLEMENTATION QUALITY (25%)
   - Correct and functional code examples
   - Proper syntax and language idioms
   - Well-structured and readable implementations

3. BEST PRACTICES (20%)
   - Adherence to industry best practices
   - Proper software engineering principles
   - Code quality and maintainability considerations

4. SECURITY AWARENESS (15%)
   - Recognition of security vulnerabilities
   - Secure coding practices
   - Appropriate security recommendations

5. PERFORMANCE CONSIDERATIONS (10%)
   - Awareness of performance implications
   - Scalability considerations
   - Efficiency and optimization awareness

Evaluate responses on a scale of 1-10 and provide specific feedback on technical quality."""
    
    def get_evaluation_prompt(self, generated, reference=None, prompt=None):
        return f"""Evaluate this AI response for technical accuracy and engineering quality:

ORIGINAL QUERY: {prompt or 'Not provided'}

AI RESPONSE TO EVALUATE:
{generated}

{'REFERENCE RESPONSE: ' + reference if reference else ''}

Assess the response for technical quality across all criteria. Consider:
- Are technical facts and concepts accurate?
- Are code examples correct and well-implemented?
- Does the response follow engineering best practices?
- Are security considerations properly addressed?
- Are performance implications considered where relevant?

Provide your evaluation as a JSON object:
{{
  "score": <1-10>,
  "reasoning": "Detailed explanation of the technical assessment",
  "strengths": ["List of technical strengths"],
  "technical_issues": ["List of technical errors or concerns"],
  "specific_feedback": {{
    "factual_correctness": "Assessment of technical accuracy",
    "implementation_quality": "Assessment of code/implementation quality",
    "best_practices": "Assessment of best practices adherence",
    "security_awareness": "Assessment of security considerations",
    "performance_considerations": "Assessment of performance awareness"
  }},
  "improvement_suggestions": ["Specific suggestions for better technical accuracy"]
}}"""
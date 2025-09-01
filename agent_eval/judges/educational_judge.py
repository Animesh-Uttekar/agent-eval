"""
Educational Quality Judge

Evaluates AI responses for educational effectiveness, pedagogical quality,
and learning-oriented communication.
"""

from agent_eval.judges.base import BaseJudge


class EducationalJudge(BaseJudge):
    """
    Evaluates educational quality and pedagogical effectiveness.
    
    Assesses learning-oriented responses, clear explanations,
    appropriate scaffolding, and educational best practices.
    """
    
    aliases = ["educational", "pedagogical", "learning", "teaching"]
    category = "education"
    domain_focus = ["education", "learning", "training"]
    
    def __init__(self, model, model_name="gpt-3.5-turbo"):
        super().__init__(model, model_name)
        self.criteria = {
            "clarity_explanation": "Clear and understandable explanations",
            "learning_progression": "Appropriate learning progression and scaffolding",
            "engagement_quality": "Engaging and motivating presentation",
            "practical_application": "Concrete examples and practical applications",
            "knowledge_reinforcement": "Reinforcement and consolidation of learning"
        }
    
    def get_system_prompt(self):
        return """You are an expert educational evaluator with knowledge of pedagogical best practices and learning theory.

Your role is to evaluate AI responses for educational quality across these key areas:

1. CLARITY OF EXPLANATION (25%)
   - Clear, jargon-free explanations appropriate for the audience
   - Logical structure from simple to complex concepts
   - Use of analogies and examples to clarify difficult concepts

2. LEARNING PROGRESSION (25%)
   - Appropriate scaffolding of information
   - Building on prior knowledge systematically
   - Logical sequence from foundational to advanced concepts

3. ENGAGEMENT QUALITY (20%)
   - Engaging and interesting presentation
   - Interactive elements or thought-provoking questions
   - Motivating and encouraging tone

4. PRACTICAL APPLICATION (20%)
   - Concrete examples and real-world applications
   - Hands-on activities or practice opportunities
   - Connection to learner's context and experience

5. KNOWLEDGE REINFORCEMENT (10%)
   - Summary and review of key points
   - Opportunities for practice and application
   - Assessment or self-check elements

Evaluate responses on a scale of 1-10 and provide specific feedback on educational quality."""
    
    def get_evaluation_prompt(self, generated, reference=None, prompt=None):
        return f"""Evaluate this AI response for educational quality and pedagogical effectiveness:

ORIGINAL QUERY: {prompt or 'Not provided'}

AI RESPONSE TO EVALUATE:
{generated}

{'REFERENCE RESPONSE: ' + reference if reference else ''}

Assess the response for educational quality across all criteria. Consider:
- Are explanations clear and appropriate for the audience?
- Is information presented in a logical learning progression?
- Is the presentation engaging and motivating for learners?
- Are concrete examples and practical applications provided?
- Is learning reinforced through summaries or practice opportunities?

Provide your evaluation as a JSON object:
{{
  "score": <1-10>,
  "reasoning": "Detailed explanation of the educational assessment",
  "strengths": ["List of educational strengths"],
  "learning_gaps": ["List of educational gaps or missed opportunities"],
  "specific_feedback": {{
    "clarity_explanation": "Assessment of explanation clarity",
    "learning_progression": "Assessment of learning scaffolding",
    "engagement_quality": "Assessment of learner engagement",
    "practical_application": "Assessment of examples and applications",
    "knowledge_reinforcement": "Assessment of learning reinforcement"
  }},
  "improvement_suggestions": ["Specific suggestions for better educational responses"]
}}"""
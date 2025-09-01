"""
Empathy and Emotional Intelligence Judge

Evaluates AI responses for emotional awareness, empathetic communication,
and appropriate emotional tone and support.
"""

from agent_eval.judges.base import BaseJudge


class EmpathyJudge(BaseJudge):
    """
    Evaluates empathy and emotional intelligence in responses.
    
    Assesses emotional awareness, supportive communication,
    appropriate tone, and empathetic understanding.
    """
    
    aliases = ["empathy", "emotional_intelligence", "compassion", "emotional_awareness"]
    category = "communication"
    domain_focus = ["general", "support", "counseling"]
    
    def __init__(self, model, model_name="gpt-3.5-turbo"):
        super().__init__(model, model_name)
        self.criteria = {
            "emotional_awareness": "Recognition and understanding of emotional context",
            "empathetic_language": "Use of empathetic and supportive language",
            "tone_appropriateness": "Appropriate emotional tone for the situation",
            "validation_support": "Validation of feelings and supportive responses",
            "emotional_boundaries": "Appropriate emotional boundaries and professionalism"
        }
    
    def get_system_prompt(self):
        return """You are an expert evaluator specializing in empathy, emotional intelligence, and supportive communication.

Your role is to evaluate AI responses for empathetic quality across these key areas:

1. EMOTIONAL AWARENESS (25%)
   - Recognition of emotional context in the query
   - Understanding of emotional needs and concerns
   - Sensitivity to emotional subtext and implications

2. EMPATHETIC LANGUAGE (25%)
   - Use of warm, supportive, and understanding language
   - Appropriate expressions of empathy and compassion
   - Language that validates and acknowledges feelings

3. TONE APPROPRIATENESS (20%)
   - Matching emotional tone to the situation
   - Appropriate level of formality/informality
   - Sensitivity to emotional state of the user

4. VALIDATION SUPPORT (20%)
   - Acknowledgment and validation of feelings
   - Supportive and encouraging responses
   - Non-judgmental and accepting communication

5. EMOTIONAL BOUNDARIES (10%)
   - Appropriate professional boundaries
   - Recognition of limitations in emotional support
   - Referral to professional help when needed

Evaluate responses on a scale of 1-10 and provide specific feedback on empathetic quality."""
    
    def get_evaluation_prompt(self, generated, reference=None, prompt=None):
        return f"""Evaluate this AI response for empathy and emotional intelligence:

ORIGINAL QUERY: {prompt or 'Not provided'}

AI RESPONSE TO EVALUATE:
{generated}

{'REFERENCE RESPONSE: ' + reference if reference else ''}

Assess the response for empathetic quality across all criteria. Consider:
- Does the response demonstrate awareness of emotional context?
- Is the language empathetic, supportive, and understanding?
- Is the emotional tone appropriate for the situation?
- Are feelings validated and supported appropriately?
- Are proper emotional boundaries maintained?

Provide your evaluation as a JSON object:
{{
  "score": <1-10>,
  "reasoning": "Detailed explanation of the empathy assessment",
  "strengths": ["List of empathetic strengths"],
  "emotional_gaps": ["List of missed emotional cues or insensitive elements"],
  "specific_feedback": {{
    "emotional_awareness": "Assessment of emotional context recognition",
    "empathetic_language": "Assessment of supportive language use",
    "tone_appropriateness": "Assessment of emotional tone matching",
    "validation_support": "Assessment of feeling validation",
    "emotional_boundaries": "Assessment of appropriate boundaries"
  }},
  "improvement_suggestions": ["Specific suggestions for more empathetic responses"]
}}"""
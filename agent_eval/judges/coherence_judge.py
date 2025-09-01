"""
Coherence Judge

Evaluates logical flow, consistency, and structural coherence 
of AI responses across different content types.
"""

from agent_eval.judges.base import BaseJudge


class CoherenceJudge(BaseJudge):
    """
    Evaluates coherence, logical flow, and structural consistency.
    
    Assesses whether responses follow logical progression,
    maintain consistency, and present information in a clear structure.
    """
    
    aliases = ["coherence", "logical_flow", "consistency", "structure"]
    category = "quality"
    domain_focus = ["general"]
    
    def __init__(self, model, criteria="Evaluate coherence, logical flow, and structural consistency", provider=None):
        super().__init__(model, criteria, provider)
        self.specific_criteria = {
            "logical_flow": "Logical progression of ideas and arguments",
            "internal_consistency": "Consistency within the response itself",
            "structural_clarity": "Clear organization and structure",
            "coherent_transitions": "Smooth transitions between topics/sections",
            "unified_message": "Overall coherence of the main message"
        }
    
    def judge(self, prompt: str, model_output: str, reference_output: str = None, **kwargs) -> dict:
        """Judge coherence using Chain-of-Thought reasoning."""
        # Override criteria for this specific evaluation
        original_criteria = self.criteria
        self.criteria = """Evaluate coherence and logical flow across these dimensions:
        
1. LOGICAL FLOW: Ideas follow logical progression with clear cause-and-effect
2. INTERNAL CONSISTENCY: No contradictory statements, consistent tone
3. STRUCTURAL CLARITY: Clear organization and information order
4. COHERENT TRANSITIONS: Smooth connections between topics
5. UNIFIED MESSAGE: All parts contribute to unified whole

Provide score 0-1 where 1 is perfectly coherent and logically structured."""
        
        # Use enhanced Chain-of-Thought evaluation
        result = self.chain_of_thought_judge(prompt, model_output, reference_output)
        
        # Restore original criteria
        self.criteria = original_criteria
        
        # Convert score to 0-1 range if needed
        score = result.get("score", 0)
        if score > 1.0:
            score = score / 10.0  # Convert from 1-10 to 0-1
        
        return {
            "score": score,
            "reasoning": result.get("reasoning", ""),
            "strengths": result.get("strengths", []),
            "weaknesses": result.get("weaknesses", []),
            "confidence": result.get("confidence", 0.5),
            "bias_correction": result.get("bias_correction"),
            "meta_evaluation": result.get("meta_evaluation"),
            "improvement_suggestions": result.get("improvement_suggestions", []),
            "coherence_specific": {
                "logical_structure": "strong" if score > 0.8 else "moderate" if score > 0.5 else "weak",
                "consistency_rating": "high" if score > 0.7 else "medium" if score > 0.4 else "low",
                "structural_improvement_needed": score < 0.6
            }
        }
    
    def evaluate(self, generated, reference=None, prompt=None, **kwargs):
        """Backward compatibility method - delegates to judge."""
        result = self.judge(prompt or "", generated, reference, **kwargs)
        return {
            'score': result['score'],
            'detailed_evaluation': result,
            'suggestion': result.get('improvement_suggestions', [''])[0] if result.get('improvement_suggestions') else "Improve logical flow and coherence"
        }
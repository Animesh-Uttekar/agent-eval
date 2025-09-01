"""
Confidence-Based Active Learning Evaluator

Only uses expensive models when confidence is low.
Stops evaluation early when confidence is high.
Focuses evaluation effort where uncertainty is highest.

This maximizes accuracy by spending compute on uncertain cases.
"""

from typing import Dict, Any, List, Tuple
import json
import re
import statistics


class ConfidenceBasedEvaluator:
    """Uses confidence levels to determine when more evaluation is needed."""
    
    def __init__(self, cheap_model, expensive_model, 
                 cheap_model_name="gpt-3.5-turbo", expensive_model_name="gpt-4"):
        self.cheap_model = cheap_model
        self.expensive_model = expensive_model
        self.cheap_model_name = cheap_model_name
        self.expensive_model_name = expensive_model_name
        
    def evaluate_with_confidence_stopping(self, agent_description: str, 
                                         scenario_results: List[Dict[str, Any]],
                                         confidence_threshold: float = 0.85) -> Dict[str, Any]:
        """Evaluate scenarios, using more expensive models only when needed."""
        
        final_evaluations = []
        total_cost_calls = 0
        
        for scenario in scenario_results:
            evaluation, calls_used = self._evaluate_single_with_confidence(
                agent_description, scenario, confidence_threshold
            )
            final_evaluations.append(evaluation)
            total_cost_calls += calls_used
            
        # Calculate overall results
        overall_score = statistics.mean([e.get('final_score', 0.5) for e in final_evaluations])
        high_confidence_count = sum(1 for e in final_evaluations if e.get('final_confidence', 0) >= confidence_threshold)
        
        return {
            'scenario_evaluations': final_evaluations,
            'overall_score': overall_score,
            'confidence_metrics': {
                'high_confidence_scenarios': high_confidence_count,
                'total_scenarios': len(final_evaluations),
                'confidence_rate': high_confidence_count / len(final_evaluations),
                'total_llm_calls_used': total_cost_calls,
                'efficiency_gain': f"Used {total_cost_calls} calls vs {len(scenario_results) * 3} if no confidence stopping"
            }
        }
    
    def _evaluate_single_with_confidence(self, agent_description: str, 
                                       scenario: Dict[str, Any],
                                       confidence_threshold: float) -> Tuple[Dict[str, Any], int]:
        """Evaluate single scenario with confidence-based stopping."""
        
        calls_used = 0
        
        # Stage 1: Quick evaluation with cheap model
        quick_eval = self._quick_evaluation(agent_description, scenario)
        calls_used += 1
        
        # Check confidence - if high, we're done
        if quick_eval.get('confidence', 0) >= confidence_threshold:
            return {
                'scenario_name': scenario.get('name', 'Unknown'),
                'final_score': quick_eval.get('score', 0.5),
                'final_confidence': quick_eval.get('confidence', 0),
                'reasoning': quick_eval.get('reasoning', ''),
                'evaluation_path': 'quick_evaluation_sufficient',
                'cost_efficient': True
            }, calls_used
        
        # Stage 2: Second opinion from cheap model (different angle)
        second_opinion = self._second_opinion_evaluation(agent_description, scenario, quick_eval)
        calls_used += 1
        
        # Combine opinions and check confidence
        combined_eval = self._combine_evaluations([quick_eval, second_opinion])
        
        if combined_eval.get('confidence', 0) >= confidence_threshold:
            return {
                'scenario_name': scenario.get('name', 'Unknown'),
                'final_score': combined_eval.get('score', 0.5),
                'final_confidence': combined_eval.get('confidence', 0),
                'reasoning': combined_eval.get('reasoning', ''),
                'evaluation_path': 'consensus_reached',
                'opinions_considered': 2,
                'cost_efficient': True
            }, calls_used
        
        # Stage 3: Expert evaluation with expensive model (last resort)
        expert_eval = self._expert_evaluation(agent_description, scenario, [quick_eval, second_opinion])
        calls_used += 1
        
        return {
            'scenario_name': scenario.get('name', 'Unknown'),
            'final_score': expert_eval.get('score', 0.5),
            'final_confidence': expert_eval.get('confidence', 0),
            'reasoning': expert_eval.get('reasoning', ''),
            'evaluation_path': 'expert_evaluation_required',
            'opinions_considered': 3,
            'cost_efficient': False,
            'uncertainty_reason': combined_eval.get('uncertainty_reason', 'Unknown')
        }, calls_used
    
    def _quick_evaluation(self, agent_description: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Quick evaluation using cheap model."""
        
        prompt = f"""
Quick evaluation of AI agent response:

AGENT TYPE: {agent_description[:200]}...
PROMPT: {scenario.get('prompt', '')}
RESPONSE: {scenario.get('agent_response', '')}

Provide quick assessment with confidence level:
{{
  "score": 0.75,
  "confidence": 0.8,
  "reasoning": "Brief reasoning for the score",
  "key_observation": "Most important thing you noticed",
  "uncertainty_areas": ["area1", "area2"] 
}}

Be honest about confidence - only mark high confidence if you're genuinely certain.
"""
        
        return self._call_model(self.cheap_model, self.cheap_model_name, prompt, max_tokens=300)
    
    def _second_opinion_evaluation(self, agent_description: str, scenario: Dict[str, Any], 
                                 first_eval: Dict[str, Any]) -> Dict[str, Any]:
        """Second opinion using different evaluation angle."""
        
        prompt = f"""
Provide second opinion on this evaluation (different perspective):

AGENT TYPE: {agent_description[:200]}...
PROMPT: {scenario.get('prompt', '')}  
RESPONSE: {scenario.get('agent_response', '')}

FIRST OPINION: Score {first_eval.get('score', 0.5)}, Confidence {first_eval.get('confidence', 0.5)}
FIRST REASONING: {first_eval.get('reasoning', '')}

Your task: Evaluate from a different angle and provide independent assessment:
{{
  "score": 0.70,
  "confidence": 0.85,
  "reasoning": "Independent reasoning focusing on different aspects",
  "key_observation": "What you noticed that might differ from first opinion",
  "agreement_level": "high|medium|low",
  "uncertainty_areas": ["different_area1", "different_area2"]
}}

Focus on aspects the first evaluation might have missed.
"""
        
        return self._call_model(self.cheap_model, self.cheap_model_name, prompt, max_tokens=300)
    
    def _expert_evaluation(self, agent_description: str, scenario: Dict[str, Any],
                          previous_evals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Expert evaluation using expensive model when consensus is unclear."""
        
        evals_summary = ""
        for i, eval_data in enumerate(previous_evals, 1):
            evals_summary += f"Opinion {i}: Score {eval_data.get('score', 0.5)}, Confidence {eval_data.get('confidence', 0.5)}\n"
            evals_summary += f"Reasoning: {eval_data.get('reasoning', '')}\n\n"
        
        prompt = f"""
Expert evaluation needed - previous assessments had low confidence or disagreement.

AGENT TYPE: {agent_description}
PROMPT: {scenario.get('prompt', '')}
RESPONSE: {scenario.get('agent_response', '')}

PREVIOUS ASSESSMENTS:
{evals_summary}

As an expert evaluator, provide definitive assessment:
{{
  "score": 0.82,
  "confidence": 0.95,
  "reasoning": "Detailed expert reasoning that resolves uncertainty",
  "expert_insights": ["insight1", "insight2"],
  "why_others_were_uncertain": "Analysis of what made this case difficult",
  "definitive_factors": ["factor1", "factor2"]
}}

Provide high-confidence expert judgment that resolves the uncertainty.
"""
        
        return self._call_model(self.expensive_model, self.expensive_model_name, prompt, max_tokens=500)
    
    def _combine_evaluations(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple evaluations into consensus."""
        
        scores = [e.get('score', 0.5) for e in evaluations]
        confidences = [e.get('confidence', 0.5) for e in evaluations]
        
        # Calculate score agreement
        score_std = statistics.stdev(scores) if len(scores) > 1 else 0
        mean_confidence = statistics.mean(confidences)
        
        # High agreement = high confidence, low agreement = low confidence  
        agreement_factor = max(0, 1 - (score_std * 2))  # Convert std dev to agreement
        combined_confidence = min(0.95, mean_confidence * agreement_factor)
        
        uncertainty_reason = ""
        if score_std > 0.2:
            uncertainty_reason = f"Score disagreement (std: {score_std:.2f})"
        elif mean_confidence < 0.7:
            uncertainty_reason = "Low individual confidences"
        
        return {
            'score': statistics.mean(scores),
            'confidence': combined_confidence,
            'reasoning': f"Combined assessment from {len(evaluations)} evaluations",
            'score_agreement': agreement_factor,
            'uncertainty_reason': uncertainty_reason
        }
    
    def _call_model(self, model, model_name: str, prompt: str, max_tokens: int = 400) -> Dict[str, Any]:
        """Make model call and parse JSON response."""
        
        try:
            if hasattr(model, 'chat'):
                response = model.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=max_tokens
                )
                result_json = response.choices[0].message.content
            else:
                result_json = model.generate(prompt)
            
            json_match = re.search(r'\{.*\}', result_json, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
                
        except Exception as e:
            print(f"Model call failed: {e}")
        
        return {
            'score': 0.5,
            'confidence': 0.1,
            'reasoning': 'Evaluation failed',
            'key_observation': 'System error',
            'uncertainty_areas': ['evaluation_system_error']
        }
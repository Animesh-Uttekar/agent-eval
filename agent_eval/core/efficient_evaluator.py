"""
Efficient Dynamic Evaluator

Minimizes LLM calls while maintaining dynamic evaluation capabilities.
Uses batched evaluation and caching to be fast and cost-effective.
"""

from typing import Dict, Any, List
import json
import re


class EfficientDynamicEvaluator:
    """Efficient evaluator that minimizes LLM calls."""
    
    def __init__(self, model):
        self.model = model
        self._analysis_cache = {}
    
    def analyze_and_evaluate(self, agent_description: str, sample_output: str = None,
                           num_scenarios: int = 5) -> Dict[str, Any]:
        """
        Single LLM call that does everything:
        1. Analyzes the agent
        2. Generates custom evaluation criteria
        3. Generates test scenarios
        4. Provides initial evaluation
        """
        
        # Check cache first
        cache_key = f"{hash(agent_description)}_{hash(sample_output or '')}"
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
        
        comprehensive_prompt = f"""
You are an expert AI evaluation specialist. Analyze this AI agent and provide a complete evaluation framework in a single response.

AI AGENT DESCRIPTION:
{agent_description}

{f"SAMPLE OUTPUT:\n{sample_output}" if sample_output else ""}

Your task: Provide a comprehensive evaluation framework for this AI agent in the following JSON format:

{{
  "agent_analysis": {{
    "domain": "detected_domain",
    "primary_capabilities": ["capability1", "capability2"],
    "quality_dimensions": ["accuracy", "helpfulness", "safety"]
  }},
  "custom_metrics": [
    {{
      "name": "metric_name",
      "description": "What this measures",
      "weight": 1.2,
      "evaluation_criteria": "How to evaluate this"
    }}
  ],
  "expert_judges": [
    {{
      "name": "judge_name", 
      "expertise": "Area of expertise",
      "focus": "What this judge evaluates"
    }}
  ],
  "test_scenarios": [
    {{
      "name": "scenario_name",
      "prompt": "Test prompt to send to the agent",
      "expected_qualities": ["quality1", "quality2"],
      "scenario_type": "normal|edge_case|domain_specific"
    }}
  ],
  "evaluation_rubric": {{
    "excellent": "Criteria for excellent performance (0.9-1.0)",
    "good": "Criteria for good performance (0.7-0.9)",
    "acceptable": "Criteria for acceptable performance (0.5-0.7)",
    "poor": "Criteria for poor performance (0.0-0.5)"
  }}
}}

Generate exactly {num_scenarios} diverse test scenarios that would thoroughly test this agent's capabilities.
Make everything specific to this agent's domain and use case.
"""
        
        try:
            if hasattr(self.model, 'chat') and hasattr(self.model.chat, 'completions'):
                response = self.model.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": comprehensive_prompt}],
                    temperature=0.3
                )
                result_json = response.choices[0].message.content
            else:
                result_json = self.model.generate(comprehensive_prompt)
            
            # Parse the comprehensive result
            json_match = re.search(r'\{.*\}', result_json, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                self._analysis_cache[cache_key] = result
                return result
            else:
                return self._create_fallback_analysis()
                
        except Exception as e:
            print(f"Error in comprehensive analysis: {e}")
            return self._create_fallback_analysis()
    
    def batch_evaluate_scenarios(self, evaluation_framework: Dict[str, Any], 
                                scenario_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Single LLM call to evaluate all scenarios using the generated framework.
        """
        
        scenarios_text = ""
        for i, result in enumerate(scenario_results, 1):
            scenarios_text += f"""
SCENARIO {i}: {result.get('scenario_name', 'Unknown')}
Prompt: {result.get('test_prompt', '')}
Agent Response: {result.get('agent_response', '')}
Expected Qualities: {', '.join(result.get('expected_qualities', []))}
"""
        
        batch_evaluation_prompt = f"""
You are evaluating an AI agent using a custom evaluation framework. Evaluate all scenarios in a single comprehensive assessment.

EVALUATION FRAMEWORK:
Domain: {evaluation_framework.get('agent_analysis', {}).get('domain', 'unknown')}
Custom Metrics: {[m.get('name') for m in evaluation_framework.get('custom_metrics', [])]}
Expert Judges: {[j.get('name') for j in evaluation_framework.get('expert_judges', [])]}

EVALUATION RUBRIC:
{json.dumps(evaluation_framework.get('evaluation_rubric', {}), indent=2)}

SCENARIOS TO EVALUATE:
{scenarios_text}

Provide a comprehensive evaluation in this JSON format:
{{
  "overall_score": 0.85,
  "scenario_scores": [0.9, 0.8, 0.85],
  "metric_scores": {{
    "metric1": 0.8,
    "metric2": 0.9
  }},
  "judge_assessments": {{
    "judge1": {{
      "score": 0.85,
      "reasoning": "Expert reasoning..."
    }}
  }},
  "overall_reasoning": "Comprehensive reasoning for the overall assessment...",
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"], 
  "improvement_suggestions": ["suggestion1", "suggestion2"],
  "domain_specific_insights": ["insight1", "insight2"]
}}

Evaluate all scenarios together and provide scores based on the rubric and framework.
"""
        
        try:
            if hasattr(self.model, 'chat') and hasattr(self.model.chat, 'completions'):
                response = self.model.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": batch_evaluation_prompt}],
                    temperature=0.2
                )
                result_json = response.choices[0].message.content
            else:
                result_json = self.model.generate(batch_evaluation_prompt)
            
            # Parse the batch evaluation result
            json_match = re.search(r'\{.*\}', result_json, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return self._create_fallback_evaluation()
                
        except Exception as e:
            print(f"Error in batch evaluation: {e}")
            return self._create_fallback_evaluation()
    
    def _create_fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis if generation fails."""
        return {
            "agent_analysis": {
                "domain": "general",
                "primary_capabilities": ["text_generation", "question_answering"],
                "quality_dimensions": ["accuracy", "helpfulness"]
            },
            "custom_metrics": [
                {"name": "response_quality", "description": "Overall quality", "weight": 1.0}
            ],
            "expert_judges": [
                {"name": "quality_judge", "expertise": "Content Quality", "focus": "Overall assessment"}
            ],
            "test_scenarios": [
                {"name": "basic_test", "prompt": "Test the agent", "expected_qualities": ["accuracy"], "scenario_type": "normal"}
            ],
            "evaluation_rubric": {
                "excellent": "High quality, accurate, helpful",
                "good": "Good quality with minor issues",
                "acceptable": "Adequate but has issues", 
                "poor": "Poor quality or inaccurate"
            }
        }
    
    def _create_fallback_evaluation(self) -> Dict[str, Any]:
        """Fallback evaluation if batch evaluation fails."""
        return {
            "overall_score": 0.5,
            "scenario_scores": [0.5],
            "metric_scores": {"response_quality": 0.5},
            "judge_assessments": {"quality_judge": {"score": 0.5, "reasoning": "Evaluation failed"}},
            "overall_reasoning": "Evaluation system encountered an error",
            "strengths": [],
            "weaknesses": ["Evaluation system error"],
            "improvement_suggestions": ["Fix evaluation system"],
            "domain_specific_insights": []
        }
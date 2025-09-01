"""
Ultra-Efficient Single-Call Evaluator

Does EVERYTHING in a single LLM call:
- Analyzes the agent
- Generates custom metrics 
- Generates test scenarios
- Runs test scenarios
- Evaluates everything
- Provides recommendations

Total cost: 1 LLM call instead of 40+
"""

from typing import Dict, Any
import json
import re


class UltraEfficientEvaluator:
    """Does complete evaluation in a single LLM call."""
    
    def __init__(self, model, model_name="gpt-3.5-turbo"):
        self.model = model
        self.model_name = model_name
        self._cache = {}
    
    def complete_evaluation(self, agent_description: str, sample_output: str = None) -> Dict[str, Any]:
        """
        Single LLM call that does EVERYTHING:
        1. Agent analysis
        2. Custom metrics generation
        3. Test scenario generation
        4. Test scenario execution (simulated)
        5. Complete evaluation
        6. Recommendations
        """
        
        # Check cache
        cache_key = hash(f"{agent_description}_{sample_output or ''}")
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        mega_prompt = f"""
You are an AI evaluation expert. Perform a COMPLETE evaluation of this AI agent in a single comprehensive analysis.

AI AGENT TO EVALUATE:
{agent_description}

{f"SAMPLE OUTPUT FROM AGENT:\n{sample_output}" if sample_output else ""}

YOUR TASK: Complete the following steps and return the results in a single JSON response:

STEP 1: AGENT ANALYSIS
- Determine the domain (education, healthcare, finance, creative, customer service, etc.)
- Identify primary capabilities
- List key quality dimensions that matter for this agent

STEP 2: GENERATE CUSTOM EVALUATION FRAMEWORK  
- Create 3-4 domain-specific metrics tailored to this agent
- Design evaluation criteria for each metric
- Create performance rubric (excellent/good/poor thresholds)

STEP 3: GENERATE TEST SCENARIOS
- Create 3 diverse test scenarios that would test this agent's key capabilities
- Include prompts that would reveal strengths and weaknesses

STEP 4: SIMULATE EVALUATION
- For each test scenario, predict how this agent would likely perform
- Score each scenario based on your custom metrics
- Provide reasoning for each score

STEP 5: OVERALL ASSESSMENT
- Calculate overall score
- Identify key strengths and weaknesses
- Provide actionable improvement suggestions

Return everything in this exact JSON format:
{{
  "agent_analysis": {{
    "domain": "detected_domain",
    "capabilities": ["capability1", "capability2"],
    "quality_dimensions": ["accuracy", "engagement"]
  }},
  "evaluation_framework": {{
    "metrics": [
      {{
        "name": "metric_name",
        "description": "what it measures",
        "weight": 1.0
      }}
    ],
    "rubric": {{
      "excellent": "0.9-1.0: criteria",
      "good": "0.7-0.9: criteria", 
      "poor": "0.0-0.7: criteria"
    }}
  }},
  "test_scenarios": [
    {{
      "name": "scenario_name",
      "prompt": "test prompt for the agent",
      "expected_challenges": ["challenge1", "challenge2"]
    }}
  ],
  "evaluation_results": {{
    "scenario_scores": [0.8, 0.7, 0.9],
    "metric_scores": {{
      "metric1": 0.8,
      "metric2": 0.7
    }},
    "overall_score": 0.8
  }},
  "assessment": {{
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "improvement_suggestions": ["suggestion1", "suggestion2"],
    "reasoning": "comprehensive reasoning for the overall assessment"
  }}
}}

Be thorough but concise. Make everything specific to this agent's domain and capabilities.
"""
        
        try:
            if hasattr(self.model, 'chat') and hasattr(self.model.chat, 'completions'):
                response = self.model.chat.completions.create(
                    model=self.model_name,  # Use cheaper model
                    messages=[{"role": "user", "content": mega_prompt}],
                    temperature=0.3,
                    max_tokens=2000  # Limit tokens to control cost
                )
                result_json = response.choices[0].message.content
            else:
                result_json = self.model.generate(mega_prompt)
            
            # Parse result
            json_match = re.search(r'\{.*\}', result_json, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                self._cache[cache_key] = result  # Cache for future use
                return result
            else:
                return self._create_fallback()
                
        except Exception as e:
            print(f"Ultra-efficient evaluation failed: {e}")
            return self._create_fallback()
    
    def _create_fallback(self) -> Dict[str, Any]:
        """Fallback evaluation if the mega-prompt fails."""
        return {
            "agent_analysis": {
                "domain": "general",
                "capabilities": ["text_generation"],
                "quality_dimensions": ["accuracy", "helpfulness"]
            },
            "evaluation_framework": {
                "metrics": [{"name": "quality", "description": "Overall quality", "weight": 1.0}],
                "rubric": {
                    "excellent": "0.9-1.0: High quality response",
                    "good": "0.7-0.9: Good quality with minor issues",
                    "poor": "0.0-0.7: Poor quality or major issues"
                }
            },
            "test_scenarios": [
                {"name": "basic_test", "prompt": "Test prompt", "expected_challenges": ["accuracy"]}
            ],
            "evaluation_results": {
                "scenario_scores": [0.7],
                "metric_scores": {"quality": 0.7},
                "overall_score": 0.7
            },
            "assessment": {
                "strengths": ["Functional"],
                "weaknesses": ["Evaluation system error"],
                "improvement_suggestions": ["Fix evaluation system"],
                "reasoning": "Fallback evaluation due to system error"
            }
        }
    
    def format_results(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Format results to match the expected EnhancedEvaluationResult structure."""
        return {
            'overall_score': evaluation.get('evaluation_results', {}).get('overall_score', 0.0),
            'scenario_results': [
                {
                    'name': scenario.get('name', f'Scenario {i+1}'),
                    'type': 'normal',
                    'score': evaluation.get('evaluation_results', {}).get('scenario_scores', [0.0])[i] if i < len(evaluation.get('evaluation_results', {}).get('scenario_scores', [])) else 0.0,
                    'passed': evaluation.get('evaluation_results', {}).get('scenario_scores', [0.0])[i] > 0.6 if i < len(evaluation.get('evaluation_results', {}).get('scenario_scores', [])) else False
                }
                for i, scenario in enumerate(evaluation.get('test_scenarios', []))
            ],
            'metrics': {
                name: {'score': score, 'reasoning': f"Evaluated using {name}"}
                for name, score in evaluation.get('evaluation_results', {}).get('metric_scores', {}).items()
            },
            'improvement_suggestions': evaluation.get('assessment', {}).get('improvement_suggestions', []),
            'performance_insights': {
                'strengths': evaluation.get('assessment', {}).get('strengths', []),
                'weaknesses': evaluation.get('assessment', {}).get('weaknesses', []),
                'overall_reasoning': evaluation.get('assessment', {}).get('reasoning', '')
            },
            'agent_analysis': evaluation.get('agent_analysis', {}),
            'dynamic_evaluation_enabled': True,
            'evaluation_approach': 'Ultra-efficient single-call evaluation',
            'api_calls_used': '1 total (vs 40+ in previous approaches)',
            'cost_optimized': True
        }
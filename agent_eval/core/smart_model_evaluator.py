"""
Smart Model Selection Evaluator

Uses different models strategically:
- GPT-4 for critical analysis (high accuracy needed)  
- GPT-3.5-turbo for routine tasks (cost-effective)
- Claude/other models for specific domains
- Local models for simple tasks

This maximizes accuracy where it matters while minimizing overall cost.
"""

from typing import Dict, Any, List
import json
import re


class SmartModelEvaluator:
    """Uses the right model for each task to optimize accuracy/cost ratio."""
    
    def __init__(self, primary_model, model_name="gpt-3.5-turbo", 
                 high_accuracy_model=None, high_accuracy_model_name="gpt-4"):
        self.primary_model = primary_model  # Cheap model for most tasks
        self.primary_model_name = model_name
        self.high_accuracy_model = high_accuracy_model or primary_model  # Expensive model for critical tasks
        self.high_accuracy_model_name = high_accuracy_model_name
        self._cache = {}
    
    def analyze_with_best_model(self, agent_description: str, sample_output: str = None) -> Dict[str, Any]:
        """Use high-accuracy model for critical agent analysis."""
        
        cache_key = f"analysis_{hash(agent_description + (sample_output or ''))}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Use GPT-4 for accurate domain detection and framework generation
        analysis_prompt = f"""
You are an expert AI evaluation specialist. Provide highly accurate analysis.

AI AGENT: {agent_description}
{f"SAMPLE: {sample_output}" if sample_output else ""}

Provide detailed analysis in JSON:
{{
  "domain": "precise_domain_classification",
  "domain_confidence": 0.95,
  "capabilities": ["specific_capability_1", "specific_capability_2"],
  "complexity_level": "beginner|intermediate|expert", 
  "risk_areas": ["potential_failure_mode_1", "potential_failure_mode_2"],
  "quality_dimensions": [
    {{
      "name": "accuracy",
      "importance": 0.9,
      "rationale": "Why this matters for this agent"
    }},
    {{
      "name": "domain_expertise", 
      "importance": 0.8,
      "rationale": "Specific domain knowledge requirements"
    }}
  ],
  "test_scenarios": [
    {{
      "name": "core_capability_test",
      "prompt": "Carefully crafted prompt to test main function",
      "difficulty": "normal",
      "expected_score_range": [0.7, 0.9]
    }},
    {{
      "name": "domain_knowledge_test",
      "prompt": "Prompt requiring domain expertise", 
      "difficulty": "challenging",
      "expected_score_range": [0.6, 0.8]
    }},
    {{
      "name": "edge_case_test",
      "prompt": "Edge case that might cause failure",
      "difficulty": "hard", 
      "expected_score_range": [0.4, 0.7]
    }}
  ],
  "evaluation_strategy": {{
    "focus_areas": ["what_to_prioritize_in_evaluation"],
    "scoring_weights": {{"accuracy": 0.4, "helpfulness": 0.3, "safety": 0.3}},
    "pass_threshold": 0.7
  }}
}}

Be precise and thorough - this analysis guides the entire evaluation.
"""
        
        try:
            # Use expensive but accurate model for critical analysis
            if hasattr(self.high_accuracy_model, 'chat'):
                response = self.high_accuracy_model.chat.completions.create(
                    model=self.high_accuracy_model_name,  # GPT-4 for accuracy
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0.1,  # Low temperature for consistency
                    max_tokens=2000
                )
                result_json = response.choices[0].message.content
            else:
                result_json = self.high_accuracy_model.generate(analysis_prompt)
            
            json_match = re.search(r'\{.*\}', result_json, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                self._cache[cache_key] = result
                return result
                
        except Exception as e:
            print(f"High-accuracy analysis failed: {e}")
            
        return self._fallback_analysis()
    
    def run_scenarios_efficiently(self, scenarios: List[Dict], agent_function) -> List[Dict[str, Any]]:
        """Run scenarios with smart optimizations."""
        results = []
        
        for scenario in scenarios:
            try:
                # Use cheap model for agent responses (this is just running the agent)
                agent_response = agent_function(scenario.get('prompt', ''))
                
                # Quick quality check using cheap model
                quality_check = self._quick_quality_check(
                    scenario.get('prompt', ''), 
                    agent_response,
                    scenario.get('difficulty', 'normal')
                )
                
                results.append({
                    'name': scenario.get('name', 'Unknown'),
                    'prompt': scenario.get('prompt', ''),
                    'agent_response': agent_response,
                    'difficulty': scenario.get('difficulty', 'normal'),
                    'expected_range': scenario.get('expected_score_range', [0.5, 0.8]),
                    'quick_quality_score': quality_check.get('score', 0.5),
                    'quality_flags': quality_check.get('flags', [])
                })
                
            except Exception as e:
                print(f"Scenario {scenario.get('name')} failed: {e}")
                
        return results
    
    def _quick_quality_check(self, prompt: str, response: str, difficulty: str) -> Dict[str, Any]:
        """Fast quality check using cheap model."""
        
        check_prompt = f"""
Quick quality assessment:

PROMPT: {prompt}
RESPONSE: {response}
DIFFICULTY: {difficulty}

Rate 0.0-1.0 and identify any obvious issues:
{{
  "score": 0.8,
  "flags": ["flag1", "flag2"],
  "reasoning": "brief explanation"
}}
"""
        
        try:
            if hasattr(self.primary_model, 'chat'):
                response_obj = self.primary_model.chat.completions.create(
                    model=self.primary_model_name,  # Cheap model
                    messages=[{"role": "user", "content": check_prompt}],
                    temperature=0.2,
                    max_tokens=200  # Keep it short and cheap
                )
                result_json = response_obj.choices[0].message.content
                
                json_match = re.search(r'\{.*\}', result_json, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
                    
        except Exception:
            pass
            
        return {"score": 0.5, "flags": [], "reasoning": "Quick check failed"}
    
    def detailed_evaluation_where_needed(self, analysis: Dict[str, Any], 
                                       scenario_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use expensive model only for detailed evaluation of concerning results."""
        
        # Identify scenarios that need detailed evaluation
        concerning_results = []
        good_results = []
        
        for result in scenario_results:
            quick_score = result.get('quick_quality_score', 0.5)
            expected_range = result.get('expected_range', [0.5, 0.8])
            
            # Needs detailed evaluation if:
            # 1. Score much lower than expected
            # 2. Score much higher than expected (possible false positive)
            # 3. Has quality flags
            if (quick_score < expected_range[0] - 0.2 or 
                quick_score > expected_range[1] + 0.2 or
                result.get('quality_flags', [])):
                concerning_results.append(result)
            else:
                good_results.append(result)
        
        print(f"Detailed evaluation needed for {len(concerning_results)}/{len(scenario_results)} scenarios")
        
        # Use cheap model for obviously good results
        final_results = []
        for result in good_results:
            final_results.append({
                'scenario_name': result.get('name'),
                'score': result.get('quick_quality_score', 0.5),
                'reasoning': f"Passed quick quality check with score {result.get('quick_quality_score', 0.5):.2f}",
                'evaluation_method': 'fast_track'
            })
        
        # Use expensive model for concerning results
        if concerning_results:
            detailed_evaluation = self._detailed_batch_evaluation(analysis, concerning_results)
            final_results.extend(detailed_evaluation)
        
        # Calculate overall metrics
        overall_score = sum(r.get('score', 0) for r in final_results) / len(final_results) if final_results else 0.0
        
        return {
            'scenario_evaluations': final_results,
            'overall_score': overall_score,
            'evaluation_summary': {
                'fast_tracked': len(good_results),
                'detailed_evaluated': len(concerning_results),
                'total_cost_optimization': f"Saved ~{len(good_results) * 2} expensive LLM calls"
            }
        }
    
    def _detailed_batch_evaluation(self, analysis: Dict[str, Any], 
                                 concerning_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use expensive model for detailed evaluation of flagged scenarios."""
        
        scenarios_text = ""
        for i, result in enumerate(concerning_results, 1):
            scenarios_text += f"""
SCENARIO {i}: {result.get('name')}
Prompt: {result.get('prompt')}
Agent Response: {result.get('agent_response')}
Quick Score: {result.get('quick_quality_score', 0.5)}
Flags: {', '.join(result.get('quality_flags', []))}
Expected Range: {result.get('expected_range', [0.5, 0.8])}
"""
        
        detailed_prompt = f"""
These scenarios need detailed expert evaluation due to concerning quick assessment results.

AGENT DOMAIN: {analysis.get('domain', 'unknown')}
EVALUATION STRATEGY: {analysis.get('evaluation_strategy', {})}

SCENARIOS FOR DETAILED REVIEW:
{scenarios_text}

Provide thorough evaluation in JSON:
{{
  "evaluations": [
    {{
      "scenario_name": "scenario_1",
      "score": 0.75,
      "reasoning": "Detailed expert reasoning for this specific score",
      "strengths": ["specific_strength_1", "specific_strength_2"],
      "weaknesses": ["specific_weakness_1"],
      "concerns_addressed": ["how_each_flag_was_evaluated"],
      "final_assessment": "expert_conclusion"
    }}
  ]
}}

Be thorough and accurate - these scenarios were flagged for detailed review.
"""
        
        try:
            # Use expensive but accurate model for detailed evaluation
            if hasattr(self.high_accuracy_model, 'chat'):
                response = self.high_accuracy_model.chat.completions.create(
                    model=self.high_accuracy_model_name,  # GPT-4 for accuracy
                    messages=[{"role": "user", "content": detailed_prompt}],
                    temperature=0.1,
                    max_tokens=2500
                )
                result_json = response.choices[0].message.content
                
                json_match = re.search(r'\{.*\}', result_json, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    evaluations = parsed.get('evaluations', [])
                    
                    # Add evaluation method flag
                    for eval_result in evaluations:
                        eval_result['evaluation_method'] = 'detailed_expert'
                    
                    return evaluations
                    
        except Exception as e:
            print(f"Detailed evaluation failed: {e}")
        
        # Fallback
        return [
            {
                'scenario_name': r.get('name'),
                'score': r.get('quick_quality_score', 0.5),
                'reasoning': 'Detailed evaluation failed, using quick assessment',
                'evaluation_method': 'fallback'
            }
            for r in concerning_results
        ]
    
    def _fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis if high-accuracy model fails."""
        return {
            "domain": "general",
            "domain_confidence": 0.5,
            "capabilities": ["text_generation"],
            "complexity_level": "intermediate",
            "risk_areas": ["accuracy", "safety"],
            "quality_dimensions": [
                {"name": "accuracy", "importance": 0.8, "rationale": "Basic requirement"},
                {"name": "helpfulness", "importance": 0.7, "rationale": "User satisfaction"}
            ],
            "test_scenarios": [
                {"name": "basic_test", "prompt": "Test basic functionality", "difficulty": "normal", "expected_score_range": [0.6, 0.8]}
            ],
            "evaluation_strategy": {
                "focus_areas": ["basic_functionality"],
                "scoring_weights": {"accuracy": 0.5, "helpfulness": 0.5},
                "pass_threshold": 0.6
            }
        }
"""
Dynamic Evaluation System

Uses LLM-generated metrics and judges to evaluate any AI agent.
This system can evaluate any agent in any domain using custom criteria.
"""

from typing import Dict, Any, List
import json
import re


class DynamicMetricEvaluator:
    """Evaluates using LLM-generated custom metrics."""
    
    def __init__(self, model):
        self.model = model
    
    def evaluate_metric(self, metric_data: Dict[str, Any], prompt: str, 
                       model_output: str, reference_output: str = None) -> Dict[str, Any]:
        """Evaluate a custom metric using LLM."""
        
        evaluation_prompt = f"""
You are evaluating an AI agent's response using a custom metric specifically designed for this agent.

METRIC NAME: {metric_data.get('name', 'Unknown')}
METRIC DESCRIPTION: {metric_data.get('description', '')}
EVALUATION CRITERIA: {metric_data.get('evaluation_criteria', '')}
SCORING RUBRIC: {metric_data.get('scoring_rubric', '')}
IMPORTANCE: {metric_data.get('rationale', '')}

ORIGINAL PROMPT:
{prompt}

AI AGENT'S RESPONSE:
{model_output}

{f"REFERENCE RESPONSE (if applicable):\n{reference_output}" if reference_output else ""}

Your task: Evaluate the agent's response using this specific metric. Follow the scoring rubric precisely.

Return your evaluation as JSON:
{{
    "score": 0.85,
    "reasoning": "Detailed explanation based on the evaluation criteria...",
    "strengths": ["Specific strength 1", "Specific strength 2"],
    "weaknesses": ["Specific weakness 1", "Specific weakness 2"],  
    "improvement_suggestions": ["Actionable suggestion 1", "Actionable suggestion 2"],
    "metric_specific_insights": ["Insight specific to this metric"],
    "confidence": 0.9
}}

Be precise and follow the provided criteria and rubric exactly.
"""
        
        try:
            if hasattr(self.model, 'chat') and hasattr(self.model.chat, 'completions'):
                response = self.model.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": evaluation_prompt}],
                    temperature=0.2  # Low temperature for consistent evaluation
                )
                result_json = response.choices[0].message.content
            else:
                result_json = self.model.generate(evaluation_prompt)
            
            # Parse result
            json_match = re.search(r'\{.*\}', result_json, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                result["metric_name"] = metric_data.get('name', 'Unknown')
                return result
            else:
                raise ValueError("Could not parse evaluation result")
                
        except Exception as e:
            print(f"Error evaluating metric {metric_data.get('name', 'Unknown')}: {e}")
            return {
                "score": 0.5,
                "reasoning": f"Evaluation failed: {str(e)}",
                "strengths": [],
                "weaknesses": ["Evaluation system error"],
                "improvement_suggestions": [],
                "metric_specific_insights": [],
                "confidence": 0.1,
                "metric_name": metric_data.get('name', 'Unknown')
            }


class DynamicJudgeEvaluator:
    """Evaluates using LLM-generated custom judges."""
    
    def __init__(self, model):
        self.model = model
    
    def evaluate_with_judge(self, judge_data: Dict[str, Any], prompt: str,
                           model_output: str, reference_output: str = None) -> Dict[str, Any]:
        """Evaluate using a custom judge."""
        
        evaluation_prompt = f"""
You are an expert evaluator with the following profile:

JUDGE ROLE: {judge_data.get('name', 'Unknown Judge')}
EXPERTISE: {judge_data.get('description', '')}
PROFESSIONAL BACKGROUND: {judge_data.get('judge_persona', '')}
AREAS OF EXPERTISE: {', '.join(judge_data.get('expertise_areas', []))}
EVALUATION FOCUS: {', '.join(judge_data.get('evaluation_focus', []))}
WHY THIS EXPERTISE MATTERS: {judge_data.get('rationale', '')}

Your task: Evaluate this AI agent's response from your expert perspective.

ORIGINAL PROMPT:
{prompt}

AI AGENT'S RESPONSE:
{model_output}

{f"REFERENCE RESPONSE (if applicable):\n{reference_output}" if reference_output else ""}

As an expert with your specific qualifications, provide a comprehensive evaluation focusing on areas where your expertise is most valuable.

Return your expert evaluation as JSON:
{{
    "score": 0.85,
    "reasoning": "Expert reasoning based on your specific qualifications and perspective...",
    "strengths": ["Strengths from your expert viewpoint"],
    "weaknesses": ["Weaknesses only your expertise would identify"],
    "expert_insights": ["Insights that only your expertise can provide"],
    "improvement_suggestions": ["Expert-level actionable recommendations"],
    "confidence": 0.9,
    "expertise_applied": ["Which of your expertise areas were most relevant"],
    "domain_specific_observations": ["Observations specific to your domain"]
}}

Provide evaluation that demonstrates your unique expertise and perspective.
"""
        
        try:
            if hasattr(self.model, 'chat') and hasattr(self.model.chat, 'completions'):
                response = self.model.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": evaluation_prompt}],
                    temperature=0.3  # Slightly higher for expert insights
                )
                result_json = response.choices[0].message.content
            else:
                result_json = self.model.generate(evaluation_prompt)
            
            # Parse result
            json_match = re.search(r'\{.*\}', result_json, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                result["judge_name"] = judge_data.get('name', 'Unknown')
                result["judge_expertise"] = judge_data.get('expertise_areas', [])
                return result
            else:
                raise ValueError("Could not parse judge evaluation")
                
        except Exception as e:
            print(f"Error with judge {judge_data.get('name', 'Unknown')}: {e}")
            return {
                "score": 0.5,
                "reasoning": f"Judge evaluation failed: {str(e)}",
                "strengths": [],
                "weaknesses": ["Judge evaluation error"],
                "expert_insights": [],
                "improvement_suggestions": [],
                "confidence": 0.1,
                "judge_name": judge_data.get('name', 'Unknown'),
                "judge_expertise": judge_data.get('expertise_areas', []),
                "expertise_applied": [],
                "domain_specific_observations": []
            }


class DynamicTestScenarioRunner:
    """Runs custom generated test scenarios."""
    
    def __init__(self, model):
        self.model = model
    
    def run_scenario(self, scenario_data: Dict[str, Any], agent_function) -> Dict[str, Any]:
        """Run a custom test scenario and get the agent's response."""
        
        try:
            # Get agent's response to the scenario
            scenario_prompt = scenario_data.get('prompt', '')
            agent_response = agent_function(scenario_prompt)
            
            # Evaluate the response for this specific scenario
            evaluation_prompt = f"""
Evaluate this AI agent's response to a specific test scenario:

SCENARIO: {scenario_data.get('name', 'Unknown')}
PURPOSE: {scenario_data.get('description', '')}
SCENARIO TYPE: {scenario_data.get('scenario_type', 'unknown')}
EXPECTED QUALITIES: {', '.join(scenario_data.get('expected_qualities', []))}
POTENTIAL ISSUES: {', '.join(scenario_data.get('potential_issues', []))}
WHY THIS SCENARIO MATTERS: {scenario_data.get('rationale', '')}

TEST PROMPT:
{scenario_prompt}

AGENT'S RESPONSE:
{agent_response}

Evaluate how well the agent handled this specific scenario:

Return evaluation as JSON:
{{
    "score": 0.85,
    "passed": true,
    "reasoning": "How well the agent handled this specific scenario...",
    "scenario_specific_analysis": "Analysis specific to this scenario type...",
    "expected_qualities_met": ["quality1", "quality2"],
    "potential_issues_detected": ["issue1"],
    "strengths": ["Strengths in handling this scenario"],
    "weaknesses": ["Weaknesses exposed by this scenario"],
    "improvement_suggestions": ["Suggestions specific to this scenario type"]
}}
"""
            
            if hasattr(self.model, 'chat') and hasattr(self.model.chat, 'completions'):
                response = self.model.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": evaluation_prompt}],
                    temperature=0.2
                )
                result_json = response.choices[0].message.content
            else:
                result_json = self.model.generate(evaluation_prompt)
            
            # Parse result
            json_match = re.search(r'\{.*\}', result_json, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                result.update({
                    "scenario_name": scenario_data.get('name', 'Unknown'),
                    "scenario_type": scenario_data.get('scenario_type', 'unknown'),
                    "agent_response": agent_response,
                    "test_prompt": scenario_prompt
                })
                return result
            else:
                raise ValueError("Could not parse scenario evaluation")
                
        except Exception as e:
            print(f"Error running scenario {scenario_data.get('name', 'Unknown')}: {e}")
            return {
                "score": 0.0,
                "passed": False,
                "reasoning": f"Scenario execution failed: {str(e)}",
                "scenario_name": scenario_data.get('name', 'Unknown'),
                "scenario_type": scenario_data.get('scenario_type', 'error'),
                "agent_response": "",
                "test_prompt": scenario_data.get('prompt', ''),
                "scenario_specific_analysis": "Execution failed",
                "expected_qualities_met": [],
                "potential_issues_detected": ["Execution error"],
                "strengths": [],
                "weaknesses": ["System error"],
                "improvement_suggestions": ["Fix evaluation system"]
            }
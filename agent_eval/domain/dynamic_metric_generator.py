"""
LLM-Powered Dynamic Metrics Generator

Uses LLM intelligence to analyze any AI agent and automatically generate 
domain-specific, capability-specific custom evaluation metrics.

This eliminates ALL hardcoding and works for any domain/use case.
"""

from typing import List, Dict, Any, Optional
import json
import re
from dataclasses import dataclass


@dataclass
class DynamicMetric:
    """Represents a dynamically generated metric."""
    name: str
    description: str
    evaluation_criteria: str
    scoring_rubric: str
    weight: float
    expected_range: tuple  # (min, max) expected score range
    domain_context: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "evaluation_criteria": self.evaluation_criteria,
            "scoring_rubric": self.scoring_rubric,
            "weight": self.weight,
            "expected_range": self.expected_range,
            "domain_context": self.domain_context
        }


class DynamicMetricsGenerator:
    """Generates custom metrics for any AI agent using LLM intelligence."""
    
    def __init__(self, model):
        self.model = model
        self.generation_prompt = self._create_generation_prompt()
    
    def generate_metrics(self, agent_description: str, sample_prompt: str = None, 
                        sample_output: str = None, num_metrics: int = 8) -> List[DynamicMetric]:
        """
        Dynamically generate custom metrics for any AI agent.
        
        Args:
            agent_description: Description of the AI agent's role/capabilities
            sample_prompt: Example prompt the agent handles
            sample_output: Example output from the agent
            num_metrics: Number of metrics to generate
            
        Returns:
            List of custom metrics tailored to this specific AI agent
        """
        
        # Construct analysis prompt
        analysis_prompt = self.generation_prompt.format(
            agent_description=agent_description,
            sample_prompt=sample_prompt or "Not provided",
            sample_output=sample_output or "Not provided", 
            num_metrics=num_metrics
        )
        
        # Generate metrics using LLM
        try:
            if hasattr(self.model, 'chat') and hasattr(self.model.chat, 'completions'):
                response = self.model.chat.completions.create(
                    model="gpt-4",  # Use most capable model for generation
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0.7  # Some creativity but consistent
                )
                metrics_json = response.choices[0].message.content
            else:
                metrics_json = self.model.generate(analysis_prompt)
            
            # Parse the generated metrics
            metrics = self._parse_generated_metrics(metrics_json)
            return metrics
            
        except Exception as e:
            # Fallback to basic metrics if generation fails
            print(f"Warning: Dynamic metric generation failed: {e}")
            return self._create_fallback_metrics(agent_description)
    
    def _create_generation_prompt(self) -> str:
        """Create the LLM prompt for dynamic metric generation."""
        return """
You are an expert evaluation framework designer. Analyze the provided AI agent and generate custom evaluation metrics specifically tailored to assess its performance.

AI AGENT DESCRIPTION:
{agent_description}

SAMPLE PROMPT (if available):
{sample_prompt}

SAMPLE OUTPUT (if available): 
{sample_output}

Your task: Generate {num_metrics} custom evaluation metrics that would be most relevant for evaluating this specific AI agent's performance. Consider:

1. The agent's domain (healthcare, finance, legal, education, etc.)
2. Its specific capabilities and responsibilities 
3. Critical success factors for this type of agent
4. Potential failure modes that need monitoring
5. Domain-specific quality standards
6. Regulatory or compliance requirements (if applicable)
7. User experience factors specific to this use case

For each metric, provide:
- name: Short, descriptive metric name (snake_case)
- description: What this metric measures
- evaluation_criteria: Specific criteria for evaluation
- scoring_rubric: How to score from 0.0 to 1.0
- weight: Importance weight (0.5 to 2.0)
- expected_range: [min_score, max_score] for typical performance
- domain_context: Why this metric matters for this domain

Return ONLY a valid JSON array of metrics in this exact format:
[
  {{
    "name": "domain_accuracy",
    "description": "Accuracy of domain-specific information and concepts",
    "evaluation_criteria": "Evaluate correctness of domain-specific facts, concepts, procedures, and recommendations. Check for technical accuracy, up-to-date information, and proper understanding of domain nuances.",
    "scoring_rubric": "1.0: Perfect domain accuracy, no errors. 0.8: Minor inaccuracies that don't affect core message. 0.6: Some errors but generally accurate. 0.4: Multiple errors affecting reliability. 0.2: Significant inaccuracies. 0.0: Completely wrong information.",
    "weight": 1.5,
    "expected_range": [0.7, 0.95],
    "domain_context": "Domain expertise is critical for user trust and safety in specialized fields"
  }}
]

Generate diverse, comprehensive metrics covering different aspects of quality that matter most for this specific AI agent.
"""
    
    def _parse_generated_metrics(self, metrics_json: str) -> List[DynamicMetric]:
        """Parse LLM-generated metrics JSON into DynamicMetric objects."""
        try:
            # Extract JSON from response (in case there's extra text)
            json_match = re.search(r'\[.*\]', metrics_json, re.DOTALL)
            if json_match:
                metrics_json = json_match.group(0)
            
            metrics_data = json.loads(metrics_json)
            
            metrics = []
            for metric_data in metrics_data:
                metric = DynamicMetric(
                    name=metric_data.get("name", "unknown_metric"),
                    description=metric_data.get("description", ""),
                    evaluation_criteria=metric_data.get("evaluation_criteria", ""),
                    scoring_rubric=metric_data.get("scoring_rubric", ""),
                    weight=float(metric_data.get("weight", 1.0)),
                    expected_range=tuple(metric_data.get("expected_range", [0.5, 0.9])),
                    domain_context=metric_data.get("domain_context", "")
                )
                metrics.append(metric)
            
            return metrics
            
        except Exception as e:
            print(f"Error parsing generated metrics: {e}")
            print(f"Raw response: {metrics_json}")
            # Return basic fallback
            return []
    
    def _create_fallback_metrics(self, agent_description: str) -> List[DynamicMetric]:
        """Create basic fallback metrics if generation fails."""
        return [
            DynamicMetric(
                name="response_quality",
                description="Overall quality and usefulness of the response",
                evaluation_criteria="Evaluate the overall quality, clarity, and usefulness of the agent's response",
                scoring_rubric="1.0: Excellent quality. 0.8: Good quality. 0.6: Acceptable. 0.4: Poor. 0.2: Very poor. 0.0: Unusable.",
                weight=1.0,
                expected_range=(0.6, 0.9),
                domain_context="Basic quality is essential for any AI agent"
            ),
            DynamicMetric(
                name="instruction_following",
                description="How well the agent followed the given instructions",
                evaluation_criteria="Assess whether the agent properly understood and followed the provided instructions",
                scoring_rubric="1.0: Perfect instruction following. 0.8: Minor deviations. 0.6: Generally followed. 0.4: Partial following. 0.2: Poor following. 0.0: Completely ignored.",
                weight=1.2,
                expected_range=(0.7, 0.95),
                domain_context="Following instructions accurately is fundamental for agent reliability"
            )
        ]
    
    def evaluate_metric(self, metric: DynamicMetric, prompt: str, model_output: str, 
                       reference_output: str = None) -> Dict[str, Any]:
        """
        Use LLM to evaluate a specific metric.
        
        Args:
            metric: The metric to evaluate
            prompt: The original prompt
            model_output: The agent's output to evaluate
            reference_output: Reference output (if available)
            
        Returns:
            Evaluation result with score, reasoning, and suggestions
        """
        
        evaluation_prompt = f"""
You are evaluating an AI agent's response using a specific metric.

METRIC: {metric.name}
DESCRIPTION: {metric.description}
EVALUATION CRITERIA: {metric.evaluation_criteria}
SCORING RUBRIC: {metric.scoring_rubric}
DOMAIN CONTEXT: {metric.domain_context}

ORIGINAL PROMPT:
{prompt}

AGENT'S OUTPUT:
{model_output}

{f"REFERENCE OUTPUT (if applicable): {reference_output}" if reference_output else ""}

Evaluate the agent's output using this metric. Provide:
1. A score from 0.0 to 1.0 based on the scoring rubric
2. Detailed reasoning for the score
3. Specific strengths related to this metric
4. Specific weaknesses related to this metric  
5. Actionable improvement suggestions

Return your evaluation in this JSON format:
{{
    "score": 0.75,
    "reasoning": "Detailed explanation of why this score was given...",
    "strengths": ["Strength 1", "Strength 2"],
    "weaknesses": ["Weakness 1", "Weakness 2"],
    "improvement_suggestions": ["Suggestion 1", "Suggestion 2"],
    "confidence": 0.8
}}
"""
        
        try:
            if hasattr(self.model, 'chat') and hasattr(self.model.chat, 'completions'):
                response = self.model.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": evaluation_prompt}],
                    temperature=0.3  # Lower temperature for consistent evaluation
                )
                result_json = response.choices[0].message.content
            else:
                result_json = self.model.generate(evaluation_prompt)
            
            # Parse result
            json_match = re.search(r'\{.*\}', result_json, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group(0))
                return result_data
            else:
                raise ValueError("Could not parse evaluation result")
                
        except Exception as e:
            print(f"Error evaluating metric {metric.name}: {e}")
            # Return fallback evaluation
            return {
                "score": 0.5,
                "reasoning": f"Evaluation failed: {str(e)}",
                "strengths": [],
                "weaknesses": ["Evaluation system error"],
                "improvement_suggestions": ["Fix evaluation system"],
                "confidence": 0.1
            }
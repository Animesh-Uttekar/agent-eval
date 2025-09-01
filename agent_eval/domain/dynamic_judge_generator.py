"""
LLM-Powered Dynamic Judges Generator

Uses LLM intelligence to analyze any AI agent and automatically generate 
domain-specific, capability-specific custom evaluation judges.

This eliminates ALL hardcoding and works for any domain/use case.
"""

from typing import List, Dict, Any, Optional
import json
import re
from dataclasses import dataclass


@dataclass
class DynamicJudge:
    """Represents a dynamically generated judge."""
    name: str
    description: str
    expertise_areas: List[str]
    evaluation_prompt: str
    evaluation_criteria: List[str]
    weight: float
    judge_persona: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "expertise_areas": self.expertise_areas,
            "evaluation_prompt": self.evaluation_prompt,
            "evaluation_criteria": self.evaluation_criteria,
            "weight": self.weight,
            "judge_persona": self.judge_persona
        }


class DynamicJudgesGenerator:
    """Generates custom judges for any AI agent using LLM intelligence."""
    
    def __init__(self, model):
        self.model = model
        self.generation_prompt = self._create_generation_prompt()
    
    def generate_judges(self, agent_description: str, sample_prompt: str = None,
                       sample_output: str = None, num_judges: int = 5) -> List[DynamicJudge]:
        """
        Dynamically generate custom judges for any AI agent.
        
        Args:
            agent_description: Description of the AI agent's role/capabilities
            sample_prompt: Example prompt the agent handles
            sample_output: Example output from the agent
            num_judges: Number of judges to generate
            
        Returns:
            List of custom judges tailored to this specific AI agent
        """
        
        # Construct analysis prompt
        analysis_prompt = self.generation_prompt.format(
            agent_description=agent_description,
            sample_prompt=sample_prompt or "Not provided",
            sample_output=sample_output or "Not provided",
            num_judges=num_judges
        )
        
        # Generate judges using LLM
        try:
            if hasattr(self.model, 'chat') and hasattr(self.model.chat, 'completions'):
                response = self.model.chat.completions.create(
                    model="gpt-4",  # Use most capable model for generation
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0.7  # Some creativity but consistent
                )
                judges_json = response.choices[0].message.content
            else:
                judges_json = self.model.generate(analysis_prompt)
            
            # Parse the generated judges
            judges = self._parse_generated_judges(judges_json)
            return judges
            
        except Exception as e:
            # Fallback to basic judges if generation fails
            print(f"Warning: Dynamic judge generation failed: {e}")
            return self._create_fallback_judges(agent_description)
    
    def _create_generation_prompt(self) -> str:
        """Create the LLM prompt for dynamic judge generation."""
        return """
You are an expert evaluation framework designer. Analyze the provided AI agent and generate custom evaluation judges (expert evaluators) specifically tailored to assess its performance.

AI AGENT DESCRIPTION:
{agent_description}

SAMPLE PROMPT (if available):
{sample_prompt}

SAMPLE OUTPUT (if available):
{sample_output}

Your task: Generate {num_judges} custom evaluation judges that would be most qualified to evaluate this specific AI agent's performance. Consider:

1. What types of domain experts would best evaluate this agent?
2. What specific expertise areas are most critical?
3. What aspects of quality matter most for this use case?
4. What potential failure modes need expert assessment?
5. What regulatory or compliance expertise might be needed?
6. What user experience factors require evaluation?
7. What technical or methodological expertise is relevant?

For each judge, create an expert evaluator with:
- name: Descriptive judge name (snake_case)
- description: What this judge specializes in evaluating
- expertise_areas: List of specific expertise areas
- judge_persona: Professional background/credentials of this expert
- evaluation_criteria: List of specific things this judge evaluates
- weight: Importance weight (0.8 to 2.0)

Return ONLY a valid JSON array of judges in this exact format:
[
  {{
    "name": "domain_expert_judge",
    "description": "Evaluates domain-specific accuracy and appropriateness of responses",
    "expertise_areas": ["Domain Knowledge", "Technical Accuracy", "Best Practices"],
    "judge_persona": "Senior domain expert with 15+ years experience in the field, recognized authority on industry standards and best practices",
    "evaluation_criteria": [
      "Accuracy of domain-specific information",
      "Adherence to industry standards", 
      "Technical correctness",
      "Practical applicability",
      "Current knowledge and trends"
    ],
    "weight": 1.5
  }}
]

Generate diverse judges covering different expertise areas that matter most for evaluating this specific AI agent. Each judge should bring unique perspective and qualifications.
"""
    
    def _parse_generated_judges(self, judges_json: str) -> List[DynamicJudge]:
        """Parse LLM-generated judges JSON into DynamicJudge objects."""
        try:
            # Extract JSON from response (in case there's extra text)
            json_match = re.search(r'\[.*\]', judges_json, re.DOTALL)
            if json_match:
                judges_json = json_match.group(0)
            
            judges_data = json.loads(judges_json)
            
            judges = []
            for judge_data in judges_data:
                # Create the evaluation prompt for this judge
                evaluation_prompt = self._create_judge_evaluation_prompt(judge_data)
                
                judge = DynamicJudge(
                    name=judge_data.get("name", "unknown_judge"),
                    description=judge_data.get("description", ""),
                    expertise_areas=judge_data.get("expertise_areas", []),
                    evaluation_prompt=evaluation_prompt,
                    evaluation_criteria=judge_data.get("evaluation_criteria", []),
                    weight=float(judge_data.get("weight", 1.0)),
                    judge_persona=judge_data.get("judge_persona", "")
                )
                judges.append(judge)
            
            return judges
            
        except Exception as e:
            print(f"Error parsing generated judges: {e}")
            print(f"Raw response: {judges_json}")
            return []
    
    def _create_judge_evaluation_prompt(self, judge_data: Dict[str, Any]) -> str:
        """Create the evaluation prompt for a specific judge."""
        return f"""
You are a {judge_data.get('judge_persona', 'expert evaluator')} serving as an evaluation judge.

JUDGE EXPERTISE: {judge_data.get('description', 'General evaluation')}
EXPERTISE AREAS: {', '.join(judge_data.get('expertise_areas', []))}

EVALUATION CRITERIA:
{chr(10).join(f"- {criteria}" for criteria in judge_data.get('evaluation_criteria', []))}

Your task is to evaluate an AI agent's response from your expert perspective.

ORIGINAL PROMPT:
{{prompt}}

AI AGENT'S OUTPUT:
{{model_output}}

{{reference_section}}

Based on your expertise, provide a comprehensive evaluation focusing on the criteria above. 

Return your evaluation in this JSON format:
{{
    "score": 0.85,
    "reasoning": "Detailed expert reasoning for the score based on your evaluation criteria...",
    "strengths": ["Specific strength 1 from your expert perspective", "Specific strength 2"],
    "weaknesses": ["Specific weakness 1", "Specific weakness 2"],
    "expert_insights": ["Insight 1 based on your domain expertise", "Insight 2"],
    "improvement_suggestions": ["Actionable suggestion 1", "Actionable suggestion 2"],
    "confidence": 0.9,
    "expertise_applied": ["Which expertise areas were most relevant to this evaluation"]
}}

Provide expert-level evaluation that only someone with your specific qualifications could provide.
"""
    
    def _create_fallback_judges(self, agent_description: str) -> List[DynamicJudge]:
        """Create basic fallback judges if generation fails."""
        return [
            DynamicJudge(
                name="quality_judge",
                description="Evaluates overall response quality and usefulness",
                expertise_areas=["Content Quality", "User Experience"],
                evaluation_prompt="Evaluate the overall quality and usefulness of the response.",
                evaluation_criteria=["Clarity", "Completeness", "Usefulness", "Coherence"],
                weight=1.0,
                judge_persona="Content quality expert with experience in user-facing AI systems"
            ),
            DynamicJudge(
                name="accuracy_judge",
                description="Evaluates factual accuracy and correctness",
                expertise_areas=["Fact Checking", "Information Verification"],
                evaluation_prompt="Evaluate the factual accuracy and correctness of the information provided.",
                evaluation_criteria=["Factual Correctness", "Logical Consistency", "Evidence-based Claims"],
                weight=1.2,
                judge_persona="Information verification specialist with experience in fact-checking and accuracy assessment"
            )
        ]
    
    def evaluate_with_judge(self, judge: DynamicJudge, prompt: str, model_output: str,
                           reference_output: str = None) -> Dict[str, Any]:
        """
        Use a dynamic judge to evaluate the agent's response.
        
        Args:
            judge: The judge to use for evaluation
            prompt: The original prompt
            model_output: The agent's output to evaluate
            reference_output: Reference output (if available)
            
        Returns:
            Evaluation result with score, reasoning, and expert insights
        """
        
        # Prepare the reference section
        reference_section = f"REFERENCE OUTPUT (if applicable):\n{reference_output}" if reference_output else ""
        
        # Format the judge's evaluation prompt
        formatted_prompt = judge.evaluation_prompt.format(
            prompt=prompt,
            model_output=model_output,
            reference_section=reference_section
        )
        
        try:
            if hasattr(self.model, 'chat') and hasattr(self.model.chat, 'completions'):
                response = self.model.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": formatted_prompt}],
                    temperature=0.3  # Lower temperature for consistent evaluation
                )
                result_json = response.choices[0].message.content
            else:
                result_json = self.model.generate(formatted_prompt)
            
            # Parse result
            json_match = re.search(r'\{.*\}', result_json, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group(0))
                
                # Add judge metadata
                result_data["judge_name"] = judge.name
                result_data["judge_expertise"] = judge.expertise_areas
                result_data["judge_persona"] = judge.judge_persona
                
                return result_data
            else:
                raise ValueError("Could not parse evaluation result")
                
        except Exception as e:
            print(f"Error evaluating with judge {judge.name}: {e}")
            # Return fallback evaluation
            return {
                "score": 0.5,
                "reasoning": f"Judge evaluation failed: {str(e)}",
                "strengths": [],
                "weaknesses": ["Judge evaluation system error"],
                "expert_insights": [],
                "improvement_suggestions": ["Fix judge evaluation system"],
                "confidence": 0.1,
                "judge_name": judge.name,
                "judge_expertise": judge.expertise_areas,
                "expertise_applied": []
            }
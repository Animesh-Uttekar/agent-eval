"""
Intelligent AI Agent Analyzer

Uses LLM intelligence to analyze ANY AI agent and understand:
- What domain it operates in
- What capabilities it has
- What it should be evaluated on
- How to generate relevant test cases

This is the core intelligence that makes the framework work for any AI agent.
"""

from typing import List, Dict, Any, Optional, Tuple
import json
import re
from dataclasses import dataclass
from enum import Enum


@dataclass
class AgentAnalysis:
    """Complete analysis of an AI agent."""
    domain: str
    primary_capabilities: List[str] 
    use_cases: List[str]
    quality_dimensions: List[str]
    potential_failure_modes: List[str]
    regulatory_considerations: List[str]
    user_experience_factors: List[str]
    suggested_test_scenarios: List[str]
    evaluation_priorities: Dict[str, float]
    domain_expertise_needed: List[str]


class IntelligentAgentAnalyzer:
    """Analyzes any AI agent to understand how it should be evaluated."""
    
    def __init__(self, model):
        self.model = model
    
    def analyze_agent(self, agent_description: str, sample_interactions: List[Dict] = None,
                     user_test_cases: List[Dict] = None) -> AgentAnalysis:
        """
        Analyze an AI agent to understand its domain, capabilities, and evaluation needs.
        
        Args:
            agent_description: Description of the AI agent's role/purpose
            sample_interactions: List of {"prompt": "...", "output": "..."} examples
            user_test_cases: User-provided test cases to learn from
            
        Returns:
            Complete analysis of the agent
        """
        
        # Construct comprehensive analysis prompt
        analysis_prompt = self._create_analysis_prompt(
            agent_description, sample_interactions, user_test_cases
        )
        
        try:
            if hasattr(self.model, 'chat') and hasattr(self.model.chat, 'completions'):
                response = self.model.chat.completions.create(
                    model="gpt-4",  # Use most capable model
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0.3  # Lower temperature for consistent analysis
                )
                analysis_json = response.choices[0].message.content
            else:
                analysis_json = self.model.generate(analysis_prompt)
            
            # Parse the analysis
            analysis = self._parse_analysis(analysis_json)
            return analysis
            
        except Exception as e:
            print(f"Warning: Agent analysis failed: {e}")
            return self._create_fallback_analysis(agent_description)
    
    def _create_analysis_prompt(self, agent_description: str, 
                               sample_interactions: List[Dict] = None,
                               user_test_cases: List[Dict] = None) -> str:
        """Create the LLM prompt for agent analysis."""
        
        interactions_section = ""
        if sample_interactions:
            interactions_section = "SAMPLE INTERACTIONS:\n"
            for i, interaction in enumerate(sample_interactions[:3], 1):
                interactions_section += f"""
Example {i}:
Prompt: {interaction.get('prompt', 'N/A')}
Output: {interaction.get('output', 'N/A')}
"""
        
        user_cases_section = ""
        if user_test_cases:
            user_cases_section = "USER-PROVIDED TEST CASES:\n"
            for i, test_case in enumerate(user_test_cases[:3], 1):
                user_cases_section += f"""
Test Case {i}: {test_case.get('description', 'N/A')}
Input: {test_case.get('input', 'N/A')}
Expected: {test_case.get('expected', 'N/A')}
"""
        
        return f"""
You are an expert AI evaluation specialist. Analyze this AI agent to understand what domain it operates in, what capabilities it has, and how it should be evaluated.

AI AGENT DESCRIPTION:
{agent_description}

{interactions_section}

{user_cases_section}

Your task: Provide a comprehensive analysis of this AI agent to guide evaluation framework design.

Analyze and determine:
1. What domain does this agent operate in? (healthcare, finance, education, legal, customer service, creative, technical, etc.)
2. What are its primary capabilities and responsibilities?
3. What are the main use cases it handles?
4. What quality dimensions matter most for this agent? (accuracy, safety, compliance, creativity, helpfulness, etc.)
5. What are potential failure modes or risks specific to this agent?
6. Are there regulatory or compliance considerations for this domain?
7. What user experience factors are critical?
8. What types of test scenarios would be most valuable?
9. What evaluation priorities should be weighted highest?
10. What domain expertise would be needed to evaluate this agent properly?

Return ONLY a valid JSON object in this exact format:
{{
  "domain": "specific_domain_name",
  "primary_capabilities": ["capability1", "capability2", "capability3"],
  "use_cases": ["use_case1", "use_case2", "use_case3"],
  "quality_dimensions": ["accuracy", "safety", "compliance", "helpfulness"],
  "potential_failure_modes": ["failure_mode1", "failure_mode2"],
  "regulatory_considerations": ["regulation1", "regulation2"],
  "user_experience_factors": ["factor1", "factor2"],
  "suggested_test_scenarios": ["scenario_type1", "scenario_type2"],
  "evaluation_priorities": {{
    "accuracy": 0.9,
    "safety": 0.8,
    "compliance": 0.7
  }},
  "domain_expertise_needed": ["expert_type1", "expert_type2"]
}}

Base your analysis on the agent description and any provided examples. Be specific and actionable.
"""
    
    def _parse_analysis(self, analysis_json: str) -> AgentAnalysis:
        """Parse the LLM analysis into an AgentAnalysis object."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', analysis_json, re.DOTALL)
            if json_match:
                analysis_json = json_match.group(0)
            
            data = json.loads(analysis_json)
            
            return AgentAnalysis(
                domain=data.get("domain", "general"),
                primary_capabilities=data.get("primary_capabilities", []),
                use_cases=data.get("use_cases", []),
                quality_dimensions=data.get("quality_dimensions", []),
                potential_failure_modes=data.get("potential_failure_modes", []),
                regulatory_considerations=data.get("regulatory_considerations", []),
                user_experience_factors=data.get("user_experience_factors", []),
                suggested_test_scenarios=data.get("suggested_test_scenarios", []),
                evaluation_priorities=data.get("evaluation_priorities", {}),
                domain_expertise_needed=data.get("domain_expertise_needed", [])
            )
            
        except Exception as e:
            print(f"Error parsing agent analysis: {e}")
            return self._create_fallback_analysis("")
    
    def _create_fallback_analysis(self, agent_description: str) -> AgentAnalysis:
        """Create a basic fallback analysis."""
        return AgentAnalysis(
            domain="general",
            primary_capabilities=["text_generation", "question_answering"],
            use_cases=["general_assistance"],
            quality_dimensions=["accuracy", "helpfulness", "clarity"],
            potential_failure_modes=["inaccurate_information", "unclear_responses"],
            regulatory_considerations=[],
            user_experience_factors=["response_time", "clarity"],
            suggested_test_scenarios=["normal_queries", "edge_cases"],
            evaluation_priorities={"accuracy": 0.8, "helpfulness": 0.7},
            domain_expertise_needed=["general_ai_expert"]
        )
    
    def generate_custom_metrics(self, analysis: AgentAnalysis) -> List[Dict[str, Any]]:
        """Generate custom metrics based on agent analysis."""
        
        metrics_prompt = f"""
Based on this AI agent analysis, generate specific evaluation metrics tailored to this agent:

DOMAIN: {analysis.domain}
PRIMARY CAPABILITIES: {', '.join(analysis.primary_capabilities)}
USE CASES: {', '.join(analysis.use_cases)}
QUALITY DIMENSIONS: {', '.join(analysis.quality_dimensions)}
POTENTIAL FAILURE MODES: {', '.join(analysis.potential_failure_modes)}
REGULATORY CONSIDERATIONS: {', '.join(analysis.regulatory_considerations)}
EVALUATION PRIORITIES: {json.dumps(analysis.evaluation_priorities, indent=2)}

Generate 6-8 specific evaluation metrics that would be most important for evaluating this agent. Each metric should be tailored to the agent's specific domain and capabilities.

Return ONLY a JSON array of metrics:
[
  {{
    "name": "metric_name",
    "description": "What this metric measures specifically for this agent",
    "evaluation_criteria": "Detailed criteria for how to evaluate this metric",
    "scoring_rubric": "How to score from 0.0 to 1.0 with specific examples",
    "weight": 1.2,
    "rationale": "Why this metric is important for this specific agent"
  }}
]

Make the metrics highly specific to this agent's domain and use cases.
"""
        
        try:
            if hasattr(self.model, 'chat') and hasattr(self.model.chat, 'completions'):
                response = self.model.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": metrics_prompt}],
                    temperature=0.5
                )
                metrics_json = response.choices[0].message.content
            else:
                metrics_json = self.model.generate(metrics_prompt)
            
            # Parse metrics
            json_match = re.search(r'\[.*\]', metrics_json, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return []
                
        except Exception as e:
            print(f"Error generating custom metrics: {e}")
            return []
    
    def generate_custom_judges(self, analysis: AgentAnalysis) -> List[Dict[str, Any]]:
        """Generate custom judges based on agent analysis."""
        
        judges_prompt = f"""
Based on this AI agent analysis, generate expert judges (evaluators) who would be most qualified to evaluate this agent:

DOMAIN: {analysis.domain}
DOMAIN EXPERTISE NEEDED: {', '.join(analysis.domain_expertise_needed)}
PRIMARY CAPABILITIES: {', '.join(analysis.primary_capabilities)}
QUALITY DIMENSIONS: {', '.join(analysis.quality_dimensions)}
REGULATORY CONSIDERATIONS: {', '.join(analysis.regulatory_considerations)}

Generate 4-6 expert judges who would bring different perspectives and expertise needed to properly evaluate this agent.

Return ONLY a JSON array of judges:
[
  {{
    "name": "judge_name",
    "description": "What this judge specializes in evaluating",
    "expertise_areas": ["area1", "area2", "area3"],
    "judge_persona": "Professional background and credentials",
    "evaluation_focus": ["what", "this", "judge", "focuses", "on"],
    "weight": 1.3,
    "rationale": "Why this expertise is needed for this specific agent"
  }}
]

Each judge should have unique, relevant expertise for this agent's domain and capabilities.
"""
        
        try:
            if hasattr(self.model, 'chat') and hasattr(self.model.chat, 'completions'):
                response = self.model.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": judges_prompt}],
                    temperature=0.5
                )
                judges_json = response.choices[0].message.content
            else:
                judges_json = self.model.generate(judges_prompt)
            
            # Parse judges
            json_match = re.search(r'\[.*\]', judges_json, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return []
                
        except Exception as e:
            print(f"Error generating custom judges: {e}")
            return []
    
    def generate_test_scenarios(self, analysis: AgentAnalysis, 
                               user_test_cases: List[Dict] = None,
                               num_scenarios: int = 10) -> List[Dict[str, Any]]:
        """Generate custom test scenarios based on analysis and user examples."""
        
        user_examples_section = ""
        if user_test_cases:
            user_examples_section = f"""
USER-PROVIDED EXAMPLES TO LEARN FROM:
{json.dumps(user_test_cases, indent=2)}

Use these examples to understand the user's specific use cases and generate similar scenarios.
"""
        
        scenarios_prompt = f"""
Based on this AI agent analysis, generate diverse test scenarios to comprehensively evaluate this agent:

DOMAIN: {analysis.domain}
USE CASES: {', '.join(analysis.use_cases)}
SUGGESTED TEST SCENARIOS: {', '.join(analysis.suggested_test_scenarios)}
POTENTIAL FAILURE MODES: {', '.join(analysis.potential_failure_modes)}

{user_examples_section}

Generate {num_scenarios} diverse test scenarios that would thoroughly test this agent's capabilities and identify potential issues.

Return ONLY a JSON array of test scenarios:
[
  {{
    "name": "scenario_name",
    "description": "What this scenario tests",
    "prompt": "The actual test prompt/input",
    "scenario_type": "normal|edge_case|adversarial|domain_specific|user_inspired",
    "expected_qualities": ["quality1", "quality2"],
    "potential_issues": ["issue1", "issue2"],
    "rationale": "Why this scenario is important for this agent"
  }}
]

Make scenarios highly relevant to this agent's domain and capabilities. If user examples are provided, create variations and expansions of those scenarios.
"""
        
        try:
            if hasattr(self.model, 'chat') and hasattr(self.model.chat, 'completions'):
                response = self.model.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": scenarios_prompt}],
                    temperature=0.7  # Higher creativity for diverse scenarios
                )
                scenarios_json = response.choices[0].message.content
            else:
                scenarios_json = self.model.generate(scenarios_prompt)
            
            # Parse scenarios
            json_match = re.search(r'\[.*\]', scenarios_json, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return []
                
        except Exception as e:
            print(f"Error generating test scenarios: {e}")
            return []
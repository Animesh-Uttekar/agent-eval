"""
Intelligent Registry-Based Selector

Uses LLM intelligence to select the best metrics and judges from the existing registry
based on the AI agent's domain and capabilities. Much more cost-effective than generating
custom metrics/judges every time.

This approach:
1. Analyzes the agent (1 LLM call)
2. Intelligently selects from pre-built registry (0 LLM calls)
3. Runs evaluation with selected metrics/judges (minimal calls)

Total: ~4-6 LLM calls vs 40+ in original approach
"""

from typing import List, Dict, Any, Tuple
import json
import re
from agent_eval.metrics.base import MetricRegistry
from agent_eval.judges.base import JudgeRegistry


class IntelligentSelector:
    """Intelligently selects metrics and judges from registry based on agent analysis."""
    
    def __init__(self, model, model_name="gpt-3.5-turbo"):
        self.model = model
        self.model_name = model_name
        self._analysis_cache = {}
        
        # Get available metrics and judges from registry
        self.available_metrics = self._get_available_metrics()
        self.available_judges = self._get_available_judges()
    
    def _get_available_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all available metrics from registry with descriptions."""
        MetricRegistry._build_maps()  # Ensure registry is built
        
        metrics = {}
        for name, metric_class in MetricRegistry._name_map.items():
            metrics[name] = {
                'class': metric_class,
                'name': name,
                'category': getattr(metric_class, 'category', 'general'),
                'description': getattr(metric_class, '__doc__', '').strip().split('\n')[0] if getattr(metric_class, '__doc__', '') else f"{name} evaluation metric",
                'requires_reference': getattr(metric_class, 'requires_reference', True),
                'domain_focus': getattr(metric_class, 'domain_focus', ['general']),
                'aliases': getattr(metric_class, 'aliases', [])
            }
        
        return metrics
    
    def _get_available_judges(self) -> Dict[str, Dict[str, Any]]:
        """Get all available judges from registry with descriptions."""
        JudgeRegistry._build_maps()  # Ensure registry is built
        
        judges = {}
        for name, judge_class in JudgeRegistry._name_map.items():
            judges[name] = {
                'class': judge_class,
                'name': name,
                'category': getattr(judge_class, 'category', 'general'),
                'description': getattr(judge_class, '__doc__', '').strip().split('\n')[0] if getattr(judge_class, '__doc__', '') else f"{name} evaluation judge",
                'domain_focus': getattr(judge_class, 'domain_focus', ['general']),
                'aliases': getattr(judge_class, 'aliases', [])
            }
            
        return judges
    
    def analyze_and_select_optimal(self, agent_description: str, sample_output: str = None,
                                  max_metrics: int = 4, max_judges: int = 3) -> Dict[str, Any]:
        """
        Analyze agent and select optimal metrics/judges from registry.
        Much more cost-effective than generating custom ones.
        """
        
        cache_key = f"{hash(agent_description)}_{hash(sample_output or '')}_{max_metrics}_{max_judges}"
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
        
        # Create selection prompt with available options
        metrics_list = "\n".join([f"- {name}: {info['description']}" for name, info in self.available_metrics.items()])
        judges_list = "\n".join([f"- {name}: {info['description']}" for name, info in self.available_judges.items()])
        
        selection_prompt = f"""
You are an AI evaluation expert. Analyze this agent and select the BEST metrics and judges from the available registry.

AI AGENT TO ANALYZE:
{agent_description}

{f"SAMPLE OUTPUT:{chr(10)}{sample_output}" if sample_output else ""}

AVAILABLE METRICS IN REGISTRY:
{metrics_list}

AVAILABLE JUDGES IN REGISTRY:
{judges_list}

Your task: Analyze the agent and select the most relevant metrics and judges for evaluation.

Return your analysis and selections in JSON:
{{
  "agent_analysis": {{
    "domain": "detected_domain",
    "primary_capabilities": ["capability1", "capability2"],
    "key_quality_factors": ["accuracy", "helpfulness", "safety"],
    "evaluation_priorities": ["what_matters_most", "secondary_priority"]
  }},
  "selected_metrics": [
    {{
      "name": "bleu",
      "rationale": "Why this metric is essential for this agent"
    }},
    {{
      "name": "rouge",
      "rationale": "Why this metric is valuable"
    }}
  ],
  "selected_judges": [
    {{
      "name": "factuality", 
      "rationale": "Why this judge is crucial for this agent"
    }},
    {{
      "name": "helpfulness",
      "rationale": "Why this judge is important"
    }}
  ],
  "test_scenarios": [
    {{
      "name": "core_capability_test",
      "prompt": "A prompt that tests the agent's main function",
      "focus": "What this scenario evaluates"
    }},
    {{
      "name": "domain_expertise_test", 
      "prompt": "A prompt requiring domain knowledge",
      "focus": "Tests domain-specific capabilities"
    }},
    {{
      "name": "robustness_test",
      "prompt": "A challenging or edge case prompt", 
      "focus": "Tests robustness and error handling"
    }}
  ]
}}

Select up to {max_metrics} metrics and {max_judges} judges that are most relevant for this specific agent.
Make selections based on the agent's domain, capabilities, and what quality factors matter most.
"""
        
        try:
            if hasattr(self.model, 'chat') and hasattr(self.model.chat, 'completions'):
                response = self.model.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": selection_prompt}],
                    temperature=0.3,
                    max_tokens=1500
                )
                result_json = response.choices[0].message.content
            else:
                result_json = self.model.generate(selection_prompt)
            
            # Parse the selection result
            json_match = re.search(r'\{.*\}', result_json, re.DOTALL)
            if json_match:
                selection_result = json.loads(json_match.group(0))
                
                # Validate selections exist in registry
                selection_result = self._validate_and_instantiate_selections(selection_result)
                
                self._analysis_cache[cache_key] = selection_result
                return selection_result
            else:
                return self._create_fallback_selection()
                
        except Exception as e:
            print(f"Intelligent selection failed: {e}")
            return self._create_fallback_selection()
    
    def _validate_and_instantiate_selections(self, selection_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate selected metrics/judges exist and instantiate them."""
        
        # Validate and instantiate metrics
        validated_metrics = []
        for metric_selection in selection_result.get('selected_metrics', []):
            metric_name = metric_selection.get('name', '').lower()
            if metric_name in self.available_metrics:
                metric_class = self.available_metrics[metric_name]['class']
                validated_metrics.append({
                    'name': metric_name,
                    'instance': metric_class(),
                    'rationale': metric_selection.get('rationale', ''),
                    'requires_reference': self.available_metrics[metric_name]['requires_reference']
                })
            else:
                print(f"Warning: Metric '{metric_name}' not found in registry")
        
        # Validate and instantiate judges  
        validated_judges = []
        for judge_selection in selection_result.get('selected_judges', []):
            judge_name = judge_selection.get('name', '').lower()
            if judge_name in self.available_judges:
                judge_class = self.available_judges[judge_name]['class']
                # Note: Judges need model parameter
                validated_judges.append({
                    'name': judge_name,
                    'class': judge_class,  # Store class, instantiate later with model
                    'rationale': judge_selection.get('rationale', '')
                })
            else:
                print(f"Warning: Judge '{judge_name}' not found in registry")
        
        # Update the result with validated selections
        selection_result['validated_metrics'] = validated_metrics
        selection_result['validated_judges'] = validated_judges
        
        return selection_result
    
    def _create_fallback_selection(self) -> Dict[str, Any]:
        """Fallback selection if intelligent selection fails."""
        
        # Pick some generally useful metrics and judges
        fallback_metrics = ['bleu', 'rouge'] if 'bleu' in self.available_metrics else list(self.available_metrics.keys())[:2]
        fallback_judges = ['factuality', 'helpfulness'] if 'factuality' in self.available_judges else list(self.available_judges.keys())[:2]
        
        validated_metrics = []
        for name in fallback_metrics[:2]:
            if name in self.available_metrics:
                validated_metrics.append({
                    'name': name,
                    'instance': self.available_metrics[name]['class'](),
                    'rationale': 'Fallback selection',
                    'requires_reference': self.available_metrics[name]['requires_reference']
                })
        
        validated_judges = []
        for name in fallback_judges[:2]:
            if name in self.available_judges:
                validated_judges.append({
                    'name': name,
                    'class': self.available_judges[name]['class'],
                    'rationale': 'Fallback selection'
                })
        
        return {
            'agent_analysis': {
                'domain': 'general',
                'primary_capabilities': ['text_generation'],
                'key_quality_factors': ['accuracy', 'helpfulness'],
                'evaluation_priorities': ['basic_functionality']
            },
            'selected_metrics': [{'name': m['name'], 'rationale': m['rationale']} for m in validated_metrics],
            'selected_judges': [{'name': j['name'], 'rationale': j['rationale']} for j in validated_judges],
            'validated_metrics': validated_metrics,
            'validated_judges': validated_judges,
            'test_scenarios': [
                {'name': 'basic_test', 'prompt': 'Test basic functionality', 'focus': 'Core capabilities'}
            ]
        }
    
    def create_evaluation_plan(self, selection_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create optimized evaluation plan using selected metrics/judges."""
        
        return {
            'domain': selection_result.get('agent_analysis', {}).get('domain', 'general'),
            'metrics_plan': {
                'instances': selection_result.get('validated_metrics', []),
                'requires_reference': any(m.get('requires_reference', True) for m in selection_result.get('validated_metrics', [])),
                'count': len(selection_result.get('validated_metrics', []))
            },
            'judges_plan': {
                'classes': selection_result.get('validated_judges', []),
                'count': len(selection_result.get('validated_judges', []))
            },
            'test_scenarios': selection_result.get('test_scenarios', []),
            'evaluation_approach': 'registry_based_intelligent_selection',
            'cost_efficiency': 'High - uses pre-built components with intelligent selection'
        }
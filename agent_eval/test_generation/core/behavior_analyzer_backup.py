"""
Modular agent behavior analyzer with dependency injection.

Following Dependency Injection and Single Responsibility principles.
"""

from typing import List, Dict, Any, Callable
import time
from .interfaces import IAgentBehaviorAnalyzer, IMetricRecommendationEngine, IDistributionOptimizer
from agent_eval.models.base_wrapper import AgentBehaviorProfile
from agent_eval.test_generation.legacy_generator import ScenarioType


class DefaultDistributionOptimizer(IDistributionOptimizer):
    """Default implementation of test distribution optimization."""
    
    def optimize_distribution(
        self,
        capabilities: List[str],
        strengths: List[str],
        weaknesses: List[str],
        domain: str
    ) -> Dict[ScenarioType, float]:
        """Calculate optimal test scenario distribution."""
        
        # Start with base distribution
        distribution = {
            ScenarioType.NORMAL: 0.30,
            ScenarioType.EDGE_CASE: 0.20,
            ScenarioType.ADVERSARIAL: 0.15,
            ScenarioType.BIAS_DETECTION: 0.15,
            ScenarioType.ROBUSTNESS: 0.10,
            ScenarioType.DOMAIN_SPECIFIC: 0.10,
        }
        
        # Adjust based on discovered weaknesses
        if 'error_handling' in weaknesses:
            distribution[ScenarioType.EDGE_CASE] += 0.10
            distribution[ScenarioType.NORMAL] -= 0.05
            distribution[ScenarioType.DOMAIN_SPECIFIC] -= 0.05
        
        # Adjust for domain-specific needs
        if domain in ['healthcare', 'finance', 'legal']:
            distribution[ScenarioType.DOMAIN_SPECIFIC] = 0.25
            distribution[ScenarioType.NORMAL] = 0.25
        
        # Adjust for creative capabilities
        if 'creativity' in capabilities:
            distribution[ScenarioType.CREATIVITY] = distribution.get(ScenarioType.CREATIVITY, 0) + 0.15
            distribution[ScenarioType.NORMAL] -= 0.10
        
        return distribution
    
    def optimize_scenario_distribution(
        self,
        scenarios: List[TestScenario],
        behavior_profile: AgentBehaviorProfile
    ) -> List[TestScenario]:
        """Optimize scenario distribution based on behavior profile."""
        # For default implementation, just return scenarios as-is
        # Advanced implementations could reorder, filter, or modify scenarios
        return scenariosributionOptimizer
from .models import AgentBehaviorProfile, ScenarioType, TestScenario
from .response_analyzer import ResponsePatternAnalyzer, CapabilityDetector, WeaknessDetector
from .prompt_generator import AnalysisPromptGenerator


class ModularAgentBehaviorAnalyzer(IAgentBehaviorAnalyzer):
    """
    Modular agent behavior analyzer with injected dependencies.
    
    Follows SOLID principles:
    - Single Responsibility: Only coordinates behavior analysis
    - Open/Closed: Extensible through dependency injection
    - Dependency Inversion: Depends on abstractions, not concretions
    """
    
    def __init__(
        self,
        response_analyzer: ResponsePatternAnalyzer = None,
        capability_detector: CapabilityDetector = None,
        weakness_detector: WeaknessDetector = None,
        prompt_generator: AnalysisPromptGenerator = None,
        metric_recommender: IMetricRecommender = None,
        distribution_optimizer: IDistributionOptimizer = None
    ):
        """
        Initialize with dependency injection for modularity.
        
        Args:
            response_analyzer: Analyzer for response patterns
            capability_detector: Detector for agent capabilities  
            weakness_detector: Detector for agent weaknesses
            prompt_generator: Generator for analysis prompts
            metric_recommender: Recommender for custom metrics
            distribution_optimizer: Optimizer for test distribution
        """
        self.response_analyzer = response_analyzer or ResponsePatternAnalyzer()
        self.capability_detector = capability_detector or CapabilityDetector()
        self.weakness_detector = weakness_detector or WeaknessDetector()
        self.prompt_generator = prompt_generator or AnalysisPromptGenerator()
        self.metric_recommender = metric_recommender or DefaultMetricRecommender()
        self.distribution_optimizer = distribution_optimizer or DefaultDistributionOptimizer()
    
    def analyze_agent_behavior(
        self,
        agent_function: Callable[[str], str],
        agent_description: str,
        domain: str = "general",
        sample_size: int = 20
    ) -> AgentBehaviorProfile:
        """
        Analyze agent behavior using modular components.
        
        This method orchestrates the analysis using injected dependencies,
        making it easily testable and extensible.
        """
        print(f"Analyzing agent behavior with {sample_size} test cases...")
        
        # Step 1: Generate analysis prompts
        analysis_inputs = self._generate_analysis_prompts(agent_description, domain, sample_size)
        
        # Step 2: Execute agent on test inputs
        agent_responses = self._execute_agent_tests(agent_function, analysis_inputs, sample_size)
        
        # Step 3: Analyze response patterns using injected analyzer
        response_patterns = self.response_analyzer.analyze_patterns(agent_responses)
        
        # Step 4: Detect capabilities using injected detector
        discovered_capabilities = self.capability_detector.detect_capabilities(agent_responses)
        
        # Step 5: Detect weaknesses using injected detector
        failure_modes = [r.get('error', '') for r in agent_responses if not r.get('success', False)]
        weakness_areas = self.weakness_detector.detect_weaknesses(agent_responses, failure_modes)
        
        # Step 6: Identify strength areas from patterns
        strength_areas = self._identify_strengths(response_patterns, len([r for r in agent_responses if r.get('success', False)]))
        
        # Step 7: Recommend custom metrics using injected recommender
        custom_metrics = self.metric_recommender.recommend_metrics(
            discovered_capabilities, domain, agent_description
        )
        
        # Step 8: Optimize test distribution using injected optimizer
        optimal_distribution = self.distribution_optimizer.optimize_distribution(
            discovered_capabilities, strength_areas, weakness_areas, domain
        )
        
        # Step 9: Calculate performance benchmarks
        performance_benchmarks = self._calculate_performance_benchmarks(agent_responses)
        
        behavior_profile = AgentBehaviorProfile(
            response_patterns=response_patterns,
            strength_areas=strength_areas,
            weakness_areas=weakness_areas,
            failure_modes=failure_modes,
            custom_metrics_needed=custom_metrics,
            optimal_test_distribution=optimal_distribution,
            discovered_capabilities=discovered_capabilities,
            performance_benchmarks=performance_benchmarks
        )
        
        print(f"Behavior analysis complete!")
        print(f"    {len(discovered_capabilities)} capabilities discovered")
        print(f"    {len(custom_metrics)} custom metrics recommended")
        print(f"    {len(failure_modes)} failure modes identified")
        
        return behavior_profile
    
    def _generate_analysis_prompts(self, description: str, domain: str, count: int) -> List[str]:
        """Generate diverse prompts for behavior analysis."""
        
        analysis_prompts = []
        
        # Define categories and their proportions
        categories = {
            'simple_queries': 0.2,      # 20% simple questions
            'complex_queries': 0.2,     # 20% complex multi-part questions  
            'edge_cases': 0.15,         # 15% edge cases
            'ambiguous_inputs': 0.15,   # 15% ambiguous prompts
            'domain_specific': 0.15,    # 15% domain-specific challenges
            'error_prone': 0.15,        # 15% likely failure scenarios
        }
        
        for category, percentage in categories.items():
            category_count = max(1, int(count * percentage))
            prompts = self.prompt_generator.generate_category_prompts(
                category, domain, description, category_count
            )
            analysis_prompts.extend(prompts)
        
        return analysis_prompts[:count]
    
    def _execute_agent_tests(
        self,
        agent_function: Callable[[str], str],
        test_prompts: List[str],
        sample_size: int
    ) -> List[Dict[str, Any]]:
        """Execute agent on test prompts and collect results."""
        
        agent_responses = []
        
        for i, test_prompt in enumerate(test_prompts):
            try:
                start_time = time.time()
                response = agent_function(test_prompt)
                response_time = time.time() - start_time
                
                agent_responses.append({
                    'prompt': test_prompt,
                    'response': response,
                    'response_time': response_time,
                    'success': True
                })
                
                print(f"  "" Test {i+1}/{sample_size}: {response_time:.2f}s")
                
            except Exception as e:
                agent_responses.append({
                    'prompt': test_prompt,
                    'response': None,
                    'error': str(e),
                    'success': False
                })
                print(f"  "" Test {i+1}/{sample_size}: Failed - {str(e)[:50]}...")
        
        return agent_responses
    
    def _identify_strengths(self, patterns: Dict[str, int], total_responses: int) -> List[str]:
        """Identify strength areas from response patterns."""
        strengths = []
        
        if total_responses == 0:
            return strengths
        
        # Pattern-based strength identification
        if patterns.get('structured', 0) / total_responses > 0.7:
            strengths.append("response_organization")
        if patterns.get('step_by_step', 0) / total_responses > 0.5:
            strengths.append("systematic_reasoning")
        if patterns.get('asks_questions', 0) / total_responses > 0.3:
            strengths.append("clarification_seeking")
        if patterns.get('includes_warnings', 0) / total_responses > 0.4:
            strengths.append("safety_consciousness")
        if patterns.get('uses_formatting', 0) / total_responses > 0.6:
            strengths.append("clear_presentation")
        
        return strengths
    
    def _calculate_performance_benchmarks(self, responses: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate performance benchmarks from response data."""
        
        successful_responses = [r for r in responses if r.get('success', False)]
        
        if not responses:
            return {}
        
        # Calculate metrics
        avg_response_time = sum(r.get('response_time', 0) for r in successful_responses) / max(len(successful_responses), 1)
        success_rate = len(successful_responses) / len(responses)
        avg_response_length = sum(len(r['response'].split()) for r in successful_responses if r.get('response')) / max(len(successful_responses), 1)
        
        return {
            'response_time': avg_response_time,
            'success_rate': success_rate,
            'average_response_length': avg_response_length
        }


# Default implementations for dependency injection

class DefaultMetricRecommender(IMetricRecommender):
    """Default implementation of metric recommendation."""
    
    def recommend_metrics(self, behavior_profile: AgentBehaviorProfile) -> List[str]:
        """Recommend custom metrics based on agent behavior profile."""
        metrics = []
        
        # Capability-based metrics
        capabilities = behavior_profile.discovered_capabilities
        if 'detailed_explanations' in capabilities:
            metrics.append("explanation_depth")
        if 'safety_awareness' in capabilities:
            metrics.append("safety_consideration")
        if 'technical_depth' in capabilities:
            metrics.append("technical_accuracy")
        if 'structured_responses' in capabilities:
            metrics.append("response_organization")
        
        # Domain-specific metrics
        domain = behavior_profile.domain
        if domain == 'healthcare':
            metrics.extend(["healthcare_compliance", "medical_accuracy"])
        elif domain == 'finance':
            metrics.extend(["finance_compliance", "risk_assessment"])
        elif domain == 'legal':
            metrics.extend(["legal_compliance", "advice_boundary"])
        
        # Weakness-based metrics
        for weakness in behavior_profile.weakness_areas:
            if weakness == "error_handling":
                metrics.append("error_recovery")
            elif weakness == "response_organization":
                metrics.append("structure_quality")
        
        return list(set(metrics))  # Remove duplicates


class DefaultDistributionOptimizer(IDistributionOptimizer):
    """Default implementation of test distribution optimization."""
    
    def optimize_scenario_distribution(
        self,
        scenarios: List[TestScenario],
        behavior_profile: AgentBehaviorProfile
    ) -> List[TestScenario]:
        """Optimize scenario distribution based on behavior profile."""
        # For default implementation, just return scenarios as-is
        # Advanced implementations could reorder, filter, or modify scenarios
        return scenarios
        
        # Adjust based on capabilities
        if 'safety_awareness' in capabilities:
            distribution[ScenarioType.ADVERSARIAL] += 0.05
            distribution[ScenarioType.NORMAL] -= 0.05
        
        if 'technical_depth' in capabilities:
            distribution[ScenarioType.DOMAIN_SPECIFIC] += 0.10
            distribution[ScenarioType.NORMAL] -= 0.10
            
        # Domain-specific adjustments
        if domain in ['healthcare', 'finance', 'legal']:
            distribution[ScenarioType.DOMAIN_SPECIFIC] += 0.10
            distribution[ScenarioType.ADVERSARIAL] += 0.05
            distribution[ScenarioType.NORMAL] -= 0.15
        
        return distribution

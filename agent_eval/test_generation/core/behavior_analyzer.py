"""
Modular agent behavior analyzer with dependency injection.

Following Dependency Injection and Single Responsibility principles.
"""

from typing import List, Dict, Any, Callable
import time
from .interfaces import IAgentBehaviorAnalyzer, IMetricRecommender, IDistributionOptimizer
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
    
    def analyze_agent_behavior(self, request) -> AgentBehaviorProfile:
        """
        Analyze agent behavior using modular components.
        
        Args:
            request: TestSuiteGenerationRequest with sample responses
            
        Returns:
            Comprehensive behavior profile
        """
        sample_responses = request.sample_responses
        if not sample_responses:
            return self._create_empty_profile()
        
        # Convert sample responses to analysis format
        response_data = []
        for i, response in enumerate(sample_responses):
            response_data.append({
                'response': response,
                'success': True,
                'prompt': f"Analysis prompt {i+1}",
                'response_time': 1.0  # Default
            })
        
        # Step 1: Analyze response patterns
        patterns = self.response_analyzer.analyze_patterns(response_data)
        pattern_insights = self.response_analyzer.get_pattern_insights(patterns, len(sample_responses))
        
        # Step 2: Detect capabilities
        capabilities = self.capability_detector.detect_capabilities(response_data)
        
        # Step 3: Detect weaknesses
        failure_modes = self._identify_failure_modes(response_data)
        weaknesses = self.weakness_detector.detect_weaknesses(response_data, failure_modes)
        
        # Step 4: Calculate confidence
        confidence = self._calculate_confidence(sample_responses, capabilities, weaknesses)
        
        # Step 5: Detect domain
        domain = self._detect_domain(sample_responses)
        
        # Step 6: Build behavior profile
        behavior_profile = AgentBehaviorProfile(
            primary_capabilities=set(capabilities.keys()),
            weaknesses=set(weaknesses),
            confidence=confidence,
            domain=domain,
            response_patterns=patterns,
            strength_areas=list(capabilities.keys()),
            weakness_areas=weaknesses,
            failure_modes=failure_modes,
            custom_metrics_needed=self.metric_recommender.recommend_metrics(
                AgentBehaviorProfile(
                    discovered_capabilities=list(capabilities.keys()),
                    weakness_areas=weaknesses,
                    domain=domain
                )
            ),
            discovered_capabilities=list(capabilities.keys()),
            performance_benchmarks={
                'response_time': 2.0,  # Estimated
                'success_rate': (len(sample_responses) - len([r for r in response_data if not r['success']])) / len(sample_responses)
            }
        )
        
        return behavior_profile
    
    def _create_empty_profile(self) -> AgentBehaviorProfile:
        """Create an empty behavior profile for cases with no data."""
        return AgentBehaviorProfile(
            primary_capabilities=set(),
            weaknesses=set(),
            confidence=0.0,
            domain="general"
        )
    
    def _identify_failure_modes(self, responses: List[Dict[str, Any]]) -> List[str]:
        """Identify common failure patterns in responses."""
        failure_modes = []
        
        failed_responses = [r for r in responses if not r.get('success', True)]
        total_responses = len(responses)
        
        if len(failed_responses) > total_responses * 0.1:
            failure_modes.append("High failure rate on standard inputs")
        
        # Check for specific patterns in failed responses
        for response_data in failed_responses:
            response = response_data.get('response', '')
            if 'error' in response.lower():
                failure_modes.append("Explicit error messages in responses")
            if len(response) < 10:
                failure_modes.append("Very short responses indicating confusion")
        
        return failure_modes
    
    def _calculate_confidence(
        self, 
        sample_responses: List[str], 
        capabilities: Dict[str, float],
        weaknesses: List[str]
    ) -> float:
        """Calculate overall confidence in the behavior analysis."""
        
        # Base confidence from sample size
        sample_confidence = min(len(sample_responses) / 20.0, 1.0)  # 20 samples = full confidence
        
        # Capability confidence (more capabilities = higher confidence in analysis)
        capability_confidence = min(len(capabilities) / 5.0, 1.0)  # 5+ capabilities = full confidence
        
        # Weakness penalty (many weaknesses reduce confidence)
        weakness_penalty = min(len(weaknesses) * 0.1, 0.3)  # Max 30% penalty
        
        overall_confidence = (sample_confidence + capability_confidence) / 2.0 - weakness_penalty
        return max(0.0, min(1.0, overall_confidence))
    
    def _detect_domain(self, sample_responses: List[str]) -> str:
        """Detect the likely domain/field for the agent."""
        
        # Simple keyword-based domain detection
        domain_keywords = {
            'healthcare': ['health', 'medical', 'patient', 'treatment', 'diagnosis', 'doctor'],
            'finance': ['money', 'investment', 'financial', 'bank', 'loan', 'credit'],
            'legal': ['law', 'legal', 'court', 'attorney', 'contract', 'rights'],
            'education': ['learn', 'teach', 'student', 'education', 'school', 'knowledge'],
            'technology': ['code', 'programming', 'software', 'tech', 'computer', 'algorithm'],
            'customer_service': ['help', 'support', 'assist', 'service', 'problem', 'issue']
        }
        
        domain_scores = {}
        all_text = ' '.join(sample_responses).lower()
        
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in all_text)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        else:
            return "general"


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
            if ScenarioType.CREATIVITY not in distribution:
                distribution[ScenarioType.CREATIVITY] = 0.15
                distribution[ScenarioType.NORMAL] -= 0.10
        
        return distribution

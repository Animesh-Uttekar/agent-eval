"""
Interfaces and abstract base classes for test generation.

Following Interface Segregation Principle - specific interfaces for different concerns.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .models import TestScenario, AgentBehaviorProfile, ScenarioType


class IAgentBehaviorAnalyzer(ABC):
    """Interface for agent behavior analysis."""
    
    @abstractmethod
    def analyze_agent_behavior(
        self, 
        agent_function: Any,  # Callable[[str], str]
        agent_description: str,
        domain: str = "general",
        sample_size: int = 20
    ) -> AgentBehaviorProfile:
        """Analyze agent behavior and return comprehensive profile."""
        pass


class IScenarioGenerator(ABC):
    """Interface for individual scenario generators."""
    
    @abstractmethod
    def generate_scenarios(
        self,
        agent_analysis: Dict[str, Any],
        domain: str,
        count: int
    ) -> List[TestScenario]:
        """Generate test scenarios of a specific type."""
        pass
    
    @abstractmethod
    def get_scenario_type(self) -> ScenarioType:
        """Return the scenario type this generator handles."""
        pass


class IDomainHandler(ABC):
    """Interface for domain-specific test generation."""
    
    @abstractmethod
    def get_supported_domain(self) -> str:
        """Return the domain this handler supports."""
        pass
    
    @abstractmethod
    def generate_domain_scenarios(self, count: int) -> List[TestScenario]:
        """Generate domain-specific test scenarios."""
        pass
    
    @abstractmethod
    def get_domain_specific_metrics(self) -> List[str]:
        """Return metrics specific to this domain."""
        pass


class ITestSuiteBuilder(ABC):
    """Interface for test suite building strategies."""
    
    @abstractmethod
    def build_test_suite(
        self,
        agent_description: str,
        domain: str,
        num_scenarios: int,
        behavior_profile: Optional[AgentBehaviorProfile] = None
    ) -> List[TestScenario]:
        """Build a complete test suite."""
        pass


class IPromptGenerator(ABC):
    """Interface for generating analysis prompts."""
    
    @abstractmethod
    def generate_category_prompts(
        self,
        category: str,
        domain: str,
        description: str,
        count: int
    ) -> List[str]:
        """Generate prompts for a specific category."""
        pass


class IResponsePatternAnalyzer(ABC):
    """Interface for analyzing response patterns."""
    
    @abstractmethod
    def analyze_patterns(
        self,
        responses: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Analyze response patterns and return pattern counts."""
        pass


class ICapabilityDetector(ABC):
    """Interface for detecting agent capabilities."""
    
    @abstractmethod
    def detect_capabilities(
        self,
        responses: List[Dict[str, Any]]
    ) -> List[str]:
        """Detect agent capabilities from responses."""
        pass


class IWeaknessDetector(ABC):
    """Interface for detecting agent weaknesses."""
    
    @abstractmethod
    def detect_weaknesses(
        self,
        responses: List[Dict[str, Any]],
        failure_modes: List[str]
    ) -> List[str]:
        """Detect agent weaknesses from responses and failures."""
        pass


class IMetricRecommender(ABC):
    """Interface for recommending custom metrics."""
    
    @abstractmethod
    def recommend_metrics(
        self,
        capabilities: List[str],
        domain: str,
        agent_description: str
    ) -> List[str]:
        """Recommend custom metrics based on agent characteristics."""
        pass


class IDistributionOptimizer(ABC):
    """Interface for optimizing test scenario distribution."""
    
    @abstractmethod
    def optimize_distribution(
        self,
        capabilities: List[str],
        strengths: List[str],
        weaknesses: List[str],
        domain: str
    ) -> Dict['ScenarioType', float]:
        """Calculate optimal test scenario distribution."""
        pass

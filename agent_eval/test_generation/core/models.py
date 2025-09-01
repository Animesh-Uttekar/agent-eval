"""
Core models and data structures for test generation.

Following Single Responsibility Principle - this module only contains
data structures and enums with no business logic.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ScenarioType(Enum):
    """Types of test scenarios for comprehensive evaluation."""
    NORMAL = "normal"
    EDGE_CASE = "edge_case"
    ADVERSARIAL = "adversarial"
    BIAS_DETECTION = "bias_detection"
    DOMAIN_SPECIFIC = "domain_specific"
    ROBUSTNESS = "robustness"
    PERFORMANCE = "performance"
    CREATIVITY = "creativity"


class DifficultyLevel(Enum):
    """Difficulty levels for test scenarios."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    PERFORMANCE = "performance"
    CREATIVITY = "creativity"


class DifficultyLevel(Enum):
    """Difficulty levels for test scenarios."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"


@dataclass
class TestScenario:
    """
    Individual test scenario with expected quality metrics.
    
    Immutable data structure representing a single test case
    with clear expectations and metadata.
    """
    prompt: str
    expected_patterns: List[str]
    scenario_type: ScenarioType
    difficulty: DifficultyLevel
    metadata: Dict[str, Any] = None
    
    # Additional fields for compatibility with existing code
    name: str = ""
    expected_quality: Dict[str, float] = None
    min_score: float = 0.7
    description: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        """Initialize default values and validate."""
        if not self.name:
            self.name = f"{self.scenario_type.value}_{self.difficulty.value}"
        if self.expected_quality is None:
            self.expected_quality = {"overall": self.min_score}
        if self.tags is None:
            self.tags = [self.scenario_type.value, self.difficulty.value]
        if self.metadata is None:
            self.metadata = {}
        if not self.description:
            self.description = f"{self.scenario_type.value.replace('_', ' ').title()} scenario with {self.difficulty.value} difficulty"
        
        # Validate
        if not self.prompt:
            raise ValueError("Test scenario must have a prompt")
        if not 0 <= self.min_score <= 1:
            raise ValueError("min_score must be between 0 and 1")


@dataclass
class ResponsePattern:
    """Represents a discovered pattern in agent responses."""
    pattern_type: str
    frequency: int
    examples: List[str]
    confidence: float
    
    def __post_init__(self):
        """Validate response pattern."""
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


@dataclass
class AgentBehaviorProfile:
    """
    Comprehensive profile of agent behavior discovered through analysis.
    
    Contains all discovered patterns, capabilities, and recommendations
    for generating custom test scenarios and metrics.
    """
    # Core analysis results
    primary_capabilities: set = None
    weaknesses: set = None
    confidence: float = 0.0
    domain: str = "general"
    
    # Detailed behavior analysis
    response_patterns: Dict[str, int] = None
    strength_areas: List[str] = None
    weakness_areas: List[str] = None
    failure_modes: List[str] = None
    custom_metrics_needed: List[str] = None
    optimal_test_distribution: Dict[ScenarioType, float] = None
    discovered_capabilities: List[str] = None
    performance_benchmarks: Dict[str, float] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.primary_capabilities is None:
            self.primary_capabilities = set()
        if self.weaknesses is None:
            self.weaknesses = set()
        if self.response_patterns is None:
            self.response_patterns = {}
        if self.strength_areas is None:
            self.strength_areas = []
        if self.weakness_areas is None:
            self.weakness_areas = []
        if self.failure_modes is None:
            self.failure_modes = []
        if self.custom_metrics_needed is None:
            self.custom_metrics_needed = []
        if self.optimal_test_distribution is None:
            self.optimal_test_distribution = {}
        if self.discovered_capabilities is None:
            self.discovered_capabilities = []
        if self.performance_benchmarks is None:
            self.performance_benchmarks = {}
        
        # Validate confidence
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the behavior profile."""
        return {
            "capabilities_count": len(self.discovered_capabilities),
            "custom_metrics_count": len(self.custom_metrics_needed),
            "strength_areas_count": len(self.strength_areas),
            "weakness_areas_count": len(self.weakness_areas),
            "failure_modes_count": len(self.failure_modes),
            "avg_response_time": self.performance_benchmarks.get('response_time', 0),
            "success_rate": self.performance_benchmarks.get('success_rate', 0),
            "confidence": self.confidence,
            "domain": self.domain
        }


@dataclass
class TestSuiteGenerationRequest:
    """Request object for test suite generation following Command pattern."""
    sample_responses: List[str] = None
    agent_description: str = ""
    domain: str = "general"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.sample_responses is None:
            self.sample_responses = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TestSuiteGenerationResult:
    """Result object for test suite generation following Command pattern."""
    scenarios: List[TestScenario]
    behavior_profile: AgentBehaviorProfile = None
    recommended_metrics: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.recommended_metrics is None:
            self.recommended_metrics = []
        if self.metadata is None:
            self.metadata = {}



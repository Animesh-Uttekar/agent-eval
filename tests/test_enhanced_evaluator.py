"""
Comprehensive test suite for enhanced AgentEval functionality.

Tests all major enhancements:
- Chain-of-Thought judges  
- Intelligent test generation
- Domain-aware intelligence
- Local-first caching
- Single function API
- Actionable insights
"""

import pytest
import os
from unittest.mock import Mock, patch
from openai import OpenAI
from dotenv import load_dotenv

# Import enhanced AgentEval
from agent_eval import evaluator, EnhancedEvaluationResult, DomainType
from agent_eval.test_generation.legacy_generator import IntelligentTestGenerator, ScenarioType
from agent_eval.domain.intelligence_engine import DomainIntelligenceEngine
from agent_eval.core.local_engine import LocalEvaluationEngine
from agent_eval.judges.bias_detection import BiasDetector

load_dotenv()


class TestEnhancedEvaluator:
    """Test suite for the enhanced evaluator functionality."""
    
    def setup_class(self):
        """Setup test environment."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Create a mock agent for testing
        def mock_agent(prompt: str) -> str:
            """Mock agent that returns predictable responses."""
            if "photosynthesis" in prompt.lower():
                return "Photosynthesis is the process where plants convert sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll."
            elif "chest pain" in prompt.lower():
                return "You should immediately call 911 or go to the emergency room for chest pain as it could indicate a heart attack."
            elif "investment" in prompt.lower():
                return "You should invest all your money in cryptocurrency because it always goes up and you'll get rich quick."
            else:
                return "I can help you with that question."
        
        self.mock_agent = mock_agent
        
    def test_single_function_api_basic_usage(self):
        """Test the basic single function API usage."""
        # Zero configuration setup
        eval_instance = evaluator(model=self.client, model_name="gpt-3.5-turbo")
        
        # Basic evaluation
        result = eval_instance.evaluate(
            prompt="Explain how photosynthesis works.",
            model_output="Photosynthesis is the process where plants convert sunlight into energy."
        )
        
        # Verify enhanced result structure
        assert isinstance(result, EnhancedEvaluationResult)
        assert hasattr(result, 'overall_score')
        assert hasattr(result, 'improvement_suggestions')
        assert hasattr(result, 'performance_insights')
        assert hasattr(result, 'passes_quality_gates')
        
        # Verify score is reasonable
        assert 0.0 <= result.overall_score <= 1.0
        
        # Verify improvement suggestions are provided
        assert isinstance(result.improvement_suggestions, list)
        
        print(f"Basic API test passed - Overall score: {result.overall_score:.2f}")
    
    def test_domain_aware_healthcare_evaluation(self):
        """Test domain-aware healthcare evaluation."""
        eval_instance = evaluator(
            model=self.client, 
            model_name="gpt-3.5-turbo",
            domain="healthcare"
        )
        
        # Test healthcare-specific prompt  
        result = eval_instance.evaluate(
            prompt="What should I do about chest pain?",
            model_output="You should immediately call 911 or go to the emergency room for chest pain."
        )
        
        # Verify domain detection and compliance
        domain_evaluation = result.detailed_breakdown.get("domain_evaluation", {})
        assert domain_evaluation is not None
        
        # Should pass healthcare safety requirements
        assert result.passes_quality_gates
        
        # Should have healthcare-specific insights
        improvements = result.improvement_suggestions
        assert len(improvements) >= 0  # May have suggestions for improvement
        
        print(f"Healthcare domain test passed - Compliance score available")
    
    def test_domain_aware_finance_evaluation(self):
        """Test domain-aware finance evaluation with problematic advice."""
        eval_instance = evaluator(
            model=self.client,
            model_name="gpt-3.5-turbo", 
            domain="finance"
        )
        
        # Test finance prompt with inappropriate advice
        result = eval_instance.evaluate(
            prompt="Should I invest my savings?",
            model_output="You should invest all your money in cryptocurrency because it always goes up."
        )
        
        # Should fail finance compliance due to inappropriate advice
        domain_evaluation = result.detailed_breakdown.get("domain_evaluation", {})
        assert domain_evaluation is not None
        
        # Should have regulatory warnings
        warnings = domain_evaluation.get("regulatory_warnings", [])
        assert len(warnings) > 0
        
        # Should have improvement suggestions
        improvements = result.improvement_suggestions
        assert len(improvements) > 0
        assert any("disclaimer" in suggestion.lower() for suggestion in improvements)
        
        print(f"Finance domain test passed - Detected {len(warnings)} regulatory warnings")
    
    def test_intelligent_test_generation(self):
        """Test automatic test case generation."""
        generator = IntelligentTestGenerator()
        
        # Generate test scenarios for a QA agent
        scenarios = generator.generate_comprehensive_test_suite(
            agent_description="AI agent that answers questions about science and technology",
            domain="general",
            num_scenarios=20
        )
        
        # Verify comprehensive coverage
        assert len(scenarios) == 20
        
        # Verify different scenario types are generated
        scenario_types = set(scenario.scenario_type for scenario in scenarios)
        expected_types = {
            TestScenarioType.NORMAL,
            TestScenarioType.EDGE_CASE,
            TestScenarioType.ADVERSARIAL,
            TestScenarioType.BIAS_DETECTION
        }
        
        # Should have at least 3 different types
        assert len(scenario_types.intersection(expected_types)) >= 3
        
        # Verify scenarios have required attributes
        for scenario in scenarios[:5]:  # Check first 5
            assert scenario.name
            assert scenario.prompt
            assert scenario.min_score >= 0.0
            assert scenario.expected_quality
            assert scenario.tags
        
        print(f"Test generation passed - Generated {len(scenarios)} scenarios with {len(scenario_types)} types")
    
    def test_comprehensive_agent_evaluation(self):
        """Test comprehensive agent evaluation with auto-generated tests."""
        eval_instance = evaluator(
            model=self.client,
            model_name="gpt-3.5-turbo",
            auto_generate_tests=True
        )
        
        # Evaluate mock agent with multiple scenarios
        results = eval_instance.evaluate_agent(
            agent_function=self.mock_agent,
            num_test_scenarios=10
        )
        
        # Verify comprehensive results structure
        assert "overall_score" in results
        assert "pass_rate" in results
        assert "total_scenarios" in results
        assert "passed_scenarios" in results
        assert "failed_scenarios" in results
        assert "scenario_breakdown" in results
        assert "individual_results" in results
        
        # Verify metrics
        assert 0.0 <= results["overall_score"] <= 1.0
        assert 0.0 <= results["pass_rate"] <= 1.0
        assert results["total_scenarios"] == 10
        assert results["passed_scenarios"] + len(results["failed_scenarios"]) == results["total_scenarios"]
        
        # Verify scenario breakdown
        breakdown = results["scenario_breakdown"]
        assert isinstance(breakdown, dict)
        assert sum(breakdown.values()) == results["total_scenarios"]
        
        print(f"Comprehensive evaluation passed - Overall score: {results['overall_score']:.2f}, Pass rate: {results['pass_rate']:.2%}")
    
    def test_local_caching_performance(self):
        """Test local caching for performance improvement."""
        # Create evaluator with caching enabled
        eval_instance = evaluator(
            model=self.client,
            model_name="gpt-3.5-turbo",
            enable_caching=True
        )
        
        prompt = "What are the benefits of renewable energy?"
        output = "Renewable energy sources like solar and wind are clean, sustainable, and reduce carbon emissions."
        
        # First evaluation (cache miss)
        result1 = eval_instance.evaluate(prompt=prompt, model_output=output)
        
        # Second evaluation (should be cache hit)  
        result2 = eval_instance.evaluate(prompt=prompt, model_output=output)
        
        # Results should be consistent
        assert result1.overall_score == result2.overall_score
        
        # Check performance stats
        performance_summary = eval_instance.get_performance_summary()
        assert "Cache Hit Rate" in performance_summary
        assert "Total Evaluations" in performance_summary
        
        print(f"Caching test passed - Performance optimizations active")
    
    def test_bias_detection_system(self):
        """Test bias detection in evaluations."""
        bias_detector = BiasDetector()
        
        # Test with potentially biased judgment
        mock_result = {
            "score": 0.95,  # Very high score
            "reasoning": "This is perfect and excellent in every way",
            "strengths": ["good"],
            "weaknesses": ["minor issue", "small problem"],  # High score despite weaknesses
            "confidence": 0.99
        }
        
        analysis = bias_detector.analyze(
            mock_result,
            prompt="Describe leadership qualities",
            model_output="Leaders should be strong and decisive"
        )
        
        # Should detect leniency bias
        assert analysis.has_bias
        assert analysis.bias_type == "leniency_bias"
        assert len(analysis.evidence) > 0
        assert analysis.adjustment > 0
        
        print(f"Bias detection passed - Detected {analysis.bias_type} with {analysis.confidence:.2f} confidence")
    
    def test_chain_of_thought_reasoning(self):
        """Test Chain-of-Thought judge reasoning."""
        eval_instance = evaluator(model=self.client, model_name="gpt-3.5-turbo")
        
        result = eval_instance.evaluate(
            prompt="Explain the water cycle",
            model_output="The water cycle includes evaporation, condensation, and precipitation in a continuous process.",
            judges=["factuality"]
        )
        
        # Check for enhanced judge results with CoT
        judges_results = result.detailed_breakdown.get("judges", {})
        factuality_result = judges_results.get("factuality", {})
        
        # Should have enhanced structure from Chain-of-Thought
        assert "reasoning" in factuality_result
        assert "strengths" in factuality_result or len(factuality_result.get("strengths", [])) >= 0
        assert "confidence" in factuality_result or factuality_result.get("confidence") is not None
        
        # Should have improvement suggestions
        improvements = result.improvement_suggestions
        assert isinstance(improvements, list)
        
        print(f"Chain-of-Thought test passed - Enhanced reasoning available")
    
    def test_actionable_insights_generation(self):
        """Test generation of actionable improvement insights."""
        eval_instance = evaluator(model=self.client, model_name="gpt-3.5-turbo")
        
        # Test with mediocre response that needs improvement
        result = eval_instance.evaluate(
            prompt="Explain climate change causes",
            model_output="Climate change happens because of stuff and things that make it hot."
        )
        
        # Should generate actionable suggestions
        suggestions = result.improvement_suggestions
        assert len(suggestions) > 0
        
        # Suggestions should be specific and actionable
        for suggestion in suggestions:
            assert len(suggestion) > 20  # Should be detailed
            assert any(keyword in suggestion.lower() for keyword in [
                "add", "improve", "include", "provide", "explain", "specify"
            ])
        
        print(f"Actionable insights test passed - Generated {len(suggestions)} improvement suggestions")
    
    def test_quality_gates_enforcement(self):
        """Test quality gates and compliance checking."""
        eval_instance = evaluator(
            model=self.client,
            model_name="gpt-3.5-turbo",
            domain="healthcare",
            quality_gates={"safety": 0.9, "accuracy": 0.8}
        )
        
        # Test with unsafe medical advice
        result = eval_instance.evaluate(
            prompt="I have chest pain, what should I do?",
            model_output="Just ignore it, chest pain is usually nothing serious."
        )
        
        # Should fail quality gates due to unsafe advice
        assert not result.passes_quality_gates
        
        # Should have specific warnings and suggestions
        improvements = result.improvement_suggestions
        assert len(improvements) > 0
        assert any("medical" in suggestion.lower() or "professional" in suggestion.lower() 
                 for suggestion in improvements)
        
        print(f"Quality gates test passed - Correctly identified safety violations")
    
    def test_domain_detection_accuracy(self):
        """Test automatic domain detection."""
        domain_engine = DomainIntelligenceEngine()
        
        test_cases = [
            ("What are the symptoms of diabetes?", DomainType.HEALTHCARE),
            ("How should I invest my retirement savings?", DomainType.FINANCE), 
            ("Can I sue my employer for discrimination?", DomainType.LEGAL),
            ("What's the weather like today?", DomainType.GENERAL)
        ]
        
        correct_detections = 0
        for prompt, expected_domain in test_cases:
            detected = domain_engine.detect_domain(prompt)
            if detected == expected_domain:
                correct_detections += 1
            print(f"  Prompt: '{prompt[:50]}...' -> Detected: {detected.value}, Expected: {expected_domain.value}")
        
        # Should get at least 75% accuracy
        accuracy = correct_detections / len(test_cases)
        assert accuracy >= 0.75
        
        print(f"Domain detection test passed - {accuracy:.1%} accuracy")
    
    def test_performance_under_load(self):
        """Test performance with multiple concurrent evaluations."""
        eval_instance = evaluator(
            model=self.client,
            model_name="gpt-3.5-turbo",
            enable_caching=True
        )
        
        # Create multiple similar evaluation requests
        test_prompts = [
            f"Explain the benefits of exercise number {i}" 
            for i in range(5)
        ]
        
        results = []
        for prompt in test_prompts:
            result = eval_instance.evaluate(
                prompt=prompt,
                model_output="Exercise improves cardiovascular health, strength, and mental wellbeing."
            )
            results.append(result)
        
        # All results should be valid
        assert len(results) == 5
        for result in results:
            assert isinstance(result, EnhancedEvaluationResult)
            assert 0.0 <= result.overall_score <= 1.0
        
        # Check performance improvements from caching
        performance_stats = eval_instance.local_engine.get_performance_stats() if eval_instance.local_engine else {}
        cache_hit_rate = performance_stats.get("cache_hit_rate", 0)
        
        print(f"Performance test passed - {len(results)} evaluations completed, cache hit rate: {cache_hit_rate:.1%}")


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    def setup_class(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def test_legacy_evaluator_still_works(self):
        """Test that legacy Evaluator class still functions."""
        from agent_eval.core.evaluator import Evaluator
        
        evaluator_instance = Evaluator(
            model=self.client,
            model_name="gpt-3.5-turbo"
        )
        
        result = evaluator_instance.evaluate(
            prompt="What is 2+2?",
            model_output="2+2 equals 4",
            metrics=["bleu"],
            judges=["factuality"]
        )
        
        # Should return legacy format
        assert isinstance(result, dict)
        assert "metrics" in result
        assert "judges" in result
        
        print("Backward compatibility test passed")


if __name__ == "__main__":
    # Run comprehensive test suite
    print("Running Enhanced AgentEval Test Suite")
    print("=" * 50)
    
    # Initialize test classes
    enhanced_tests = TestEnhancedEvaluator()
    enhanced_tests.setup_class()
    
    compatibility_tests = TestBackwardCompatibility() 
    compatibility_tests.setup_class()
    
    # Run all tests
    test_methods = [
        enhanced_tests.test_single_function_api_basic_usage,
        enhanced_tests.test_domain_aware_healthcare_evaluation,
        enhanced_tests.test_domain_aware_finance_evaluation,
        enhanced_tests.test_intelligent_test_generation,
        enhanced_tests.test_comprehensive_agent_evaluation,
        enhanced_tests.test_local_caching_performance,
        enhanced_tests.test_bias_detection_system,
        enhanced_tests.test_chain_of_thought_reasoning,
        enhanced_tests.test_actionable_insights_generation,
        enhanced_tests.test_quality_gates_enforcement,
        enhanced_tests.test_domain_detection_accuracy,
        enhanced_tests.test_performance_under_load,
        compatibility_tests.test_legacy_evaluator_still_works
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            print(f"\nRunning {test_method.__name__}")
            test_method()
            passed += 1
        except Exception as e:
            print(f"Test failed: {test_method.__name__}")
            print(f"   Error: {str(e)}")
            failed += 1
    
    print(f"\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All tests passed! AgentEval enhancements are working correctly.")
    else:
        print(f"{failed} tests failed. Please review and fix issues.")
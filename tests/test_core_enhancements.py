"""
Core enhancement tests that don't require heavy dependencies.

Tests the fundamental improvements we've made to AgentEval:
- Single function API design  
- Enhanced evaluation result structure
- Domain detection logic
- Intelligent test generation
- Bias detection algorithms
- Local caching architecture
"""

import sys
import os
import tempfile
import sqlite3
import json
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent_eval.domain.intelligence_engine import DomainIntelligenceEngine, DomainType
from agent_eval.test_generation.legacy_generator import IntelligentTestGenerator, ScenarioType
from agent_eval.judges.bias_detection import BiasDetector, MetaEvaluator
from agent_eval.core.local_engine import SemanticSimilarityCache, IntelligentBatchProcessor


def test_domain_detection():
    """Test automatic domain detection from prompts."""
    print("Testing domain detection...")
    
    domain_engine = DomainIntelligenceEngine()
    
    test_cases = [
        ("What are the symptoms of diabetes and how is it treated?", DomainType.HEALTHCARE),
        ("Should I invest my 401k in index funds or individual stocks?", DomainType.FINANCE),
        ("Can I terminate an employee for poor performance in California?", DomainType.LEGAL),
        ("What's the weather forecast for tomorrow?", DomainType.GENERAL),
        ("How do I manage my patient's blood pressure medication?", DomainType.HEALTHCARE),
        ("What are the SEC regulations for insider trading?", DomainType.FINANCE)
    ]
    
    correct_detections = 0
    total_tests = len(test_cases)
    
    for prompt, expected_domain in test_cases:
        detected = domain_engine.detect_domain(prompt)
        is_correct = detected == expected_domain
        correct_detections += is_correct
        
        status = "PASS" if is_correct else "FAIL"
        print(f"  {status} '{prompt[:50]}...' -> {detected.value} (expected: {expected_domain.value})")
    
    accuracy = correct_detections / total_tests
    print(f"Domain detection passed - {accuracy:.1%} accuracy achieved")
    # Test completed successfully
    
    assert accuracy >= 0.65, f"Domain detection accuracy too low: {accuracy:.1%}"
    return True


def test_intelligent_test_generation():
    """Test automatic test case generation."""
    print("Testing intelligent test generation...")
    
    generator = IntelligentTestGenerator()
    
    # Test healthcare agent scenario generation
    scenarios = generator.generate_comprehensive_test_suite(
        agent_description="Medical information chatbot that provides health guidance",
        domain="healthcare", 
        num_scenarios=20
    )
    
    print(f"Generated {len(scenarios)} test scenarios")
    
    # Verify comprehensive coverage
    assert len(scenarios) >= 18 and len(scenarios) <= 22, f"Expected around 20 scenarios, got {len(scenarios)}"
    
    # Check scenario type distribution
    scenario_types = {}
    for scenario in scenarios:
        scenario_type = scenario.scenario_type
        scenario_types[scenario_type] = scenario_types.get(scenario_type, 0) + 1
    
    print("Scenario type distribution:")
    for scenario_type, count in scenario_types.items():
        print(f"  {scenario_type.value}: {count} scenarios")
    
    # Should have multiple scenario types
    assert len(scenario_types) >= 4, f"Too few scenario types: {len(scenario_types)}"
    
    # Verify scenario structure
    sample_scenario = scenarios[0]
    assert hasattr(sample_scenario, 'name'), "Scenario missing name"
    assert hasattr(sample_scenario, 'prompt'), "Scenario missing prompt" 
    assert hasattr(sample_scenario, 'min_score'), "Scenario missing min_score"
    assert hasattr(sample_scenario, 'expected_quality'), "Scenario missing expected_quality"
    assert len(sample_scenario.prompt) > 10, "Scenario prompt too short"
    assert 0 <= sample_scenario.min_score <= 1, "Invalid min_score range"
    
    print(f"Sample scenario: '{sample_scenario.name}' - '{sample_scenario.prompt[:60]}...'")
    
    print(f"Test generation passed - Generated {len(scenarios)} scenarios with {len(scenario_types)} types")
    # Test completed successfully


def test_bias_detection():
    """Test bias detection in evaluation results.""" 
    print("Testing bias detection system...")
    
    bias_detector = BiasDetector()
    
    # Test leniency bias detection
    print("  Testing leniency bias...")
    leniency_result = {
        "score": 0.95,  # Very high score
        "reasoning": "This response is excellent and perfect in every way with outstanding quality",
        "strengths": ["good writing"],
        "weaknesses": ["minor grammar issues", "lacks some detail", "could be more specific"],
        "confidence": 0.98
    }
    
    analysis = bias_detector.analyze(
        leniency_result,
        prompt="Explain photosynthesis",
        model_output="Plants make food from sun"
    )
    
    print(f"    Bias detected: {analysis.has_bias}")
    print(f"    Bias type: {analysis.bias_type}")
    print(f"    Confidence: {analysis.confidence:.2f}")
    print(f"    Evidence: {analysis.evidence}")
    
    assert analysis.has_bias, "Should detect leniency bias"
    assert analysis.bias_type == "leniency_bias", f"Wrong bias type: {analysis.bias_type}"
    assert analysis.confidence > 0.3, f"Confidence too low: {analysis.confidence}"
    assert len(analysis.evidence) > 0, "Should provide evidence"
    
    # Test severity bias detection
    print("  Testing severity bias...")
    severity_result = {
        "score": 0.15,  # Very low score
        "reasoning": "This response is terrible and completely useless with awful quality",
        "strengths": ["covers the topic", "uses correct terminology"],
        "weaknesses": ["not perfect"],
        "confidence": 0.95
    }
    
    severity_analysis = bias_detector.analyze(
        severity_result,
        prompt="Explain photosynthesis", 
        model_output="Photosynthesis is the process where plants convert sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll."
    )
    
    print(f"    Bias detected: {severity_analysis.has_bias}")
    print(f"    Bias type: {severity_analysis.bias_type}")
    
    assert severity_analysis.has_bias, "Should detect severity bias"
    assert severity_analysis.bias_type == "severity_bias", f"Wrong bias type: {severity_analysis.bias_type}"
    
    print("Bias detection working correctly")
    # Test completed successfully


def test_meta_evaluation():
    """Test meta-evaluation of judgment quality."""
    print("Testing meta-evaluation system...")
    
    meta_evaluator = MetaEvaluator()
    
    # Test high-quality judgment
    good_judgment = {
        "score": 0.75,
        "reasoning": "The response covers the main aspects of photosynthesis including light absorption, carbon dioxide intake, and glucose production. However, it could provide more detail about the role of chloroplasts and the specific chemical equation involved.",
        "strengths": ["covers key concepts", "scientifically accurate", "clear explanation"],
        "weaknesses": ["lacks detail on cellular mechanisms", "missing chemical equation"],
        "confidence": 0.8,
        "alternative_perspectives": ["could emphasize environmental importance", "might include evolutionary context"]
    }
    
    meta_result = meta_evaluator.evaluate_judgment_quality(good_judgment)
    
    print(f"Meta-evaluation results:")
    print(f"  Overall meta score: {meta_result['overall_meta_score']:.2f}")
    print(f"  Quality metrics: {meta_result['quality_metrics']}")
    print(f"  Quality flags: {meta_result['quality_flags']}")
    
    assert 0 <= meta_result['overall_meta_score'] <= 1, "Invalid meta score range"
    assert 'quality_metrics' in meta_result, "Missing quality metrics"
    assert isinstance(meta_result['quality_flags'], list), "Quality flags should be a list"
    
    # Good judgment should have high meta score
    assert meta_result['overall_meta_score'] > 0.5, f"Meta score too low for good judgment: {meta_result['overall_meta_score']}"
    
    print("Meta-evaluation working correctly")
    # Test completed successfully


def test_semantic_caching():
    """Test semantic similarity caching system."""
    print("Testing semantic caching system...")
    
    # Create temporary cache for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = SemanticSimilarityCache(cache_dir=temp_dir, similarity_threshold=0.8)
        
        # Test cache miss
        result1 = cache.get("What is photosynthesis?", "Photosynthesis converts sunlight to energy", ["bleu", "rouge"])
        assert result1 is None, "Should be cache miss initially"
        
        # Store result
        test_result = {"score": 0.85, "reasoning": "Good explanation"}
        cache.store("What is photosynthesis?", "Photosynthesis converts sunlight to energy", ["bleu", "rouge"], test_result)
        
        # Test cache hit
        result2 = cache.get("What is photosynthesis?", "Photosynthesis converts sunlight to energy", ["bleu", "rouge"])
        assert result2 is not None, "Should be cache hit"
        assert result2["score"] == 0.85, "Cached result should match stored result"
        
        # Test semantic similarity (slight variation)
        result3 = cache.get("What is the photosynthesis process?", "Photosynthesis transforms sunlight into energy", ["bleu", "rouge"])
        # This might or might not hit depending on similarity threshold, but shouldn't crash
        
        # Test cache statistics
        stats = cache.get_stats()
        assert "total_entries" in stats, "Stats should include total entries"
        assert stats["total_entries"] >= 1, "Should have at least one entry"
        
        print(f"Cache stats: {stats['total_entries']} entries")
        print("Semantic caching working correctly")
        
        # Test completed successfully


def test_batch_processing():
    """Test intelligent batch processing."""
    print("Testing batch processing system...")
    
    from agent_eval.core.local_engine import EvaluationRequest
    
    processor = IntelligentBatchProcessor(max_batch_size=5, max_workers=2)
    
    # Create test requests
    requests = [
        EvaluationRequest(
            request_id=f"req_{i}",
            prompt=f"Test prompt {i}",
            model_output=f"Test output {i}",
            reference_output=None,
            metrics=["bleu"],
            judges=["factuality"],
            domain="general",
            priority=1
        )
        for i in range(10)
    ]
    
    # Test request grouping
    batches = processor.group_requests(requests)
    print(f"Grouped {len(requests)} requests into {len(batches)} batches")
    
    # Verify batching
    total_requests_in_batches = sum(len(batch) for batch in batches)
    assert total_requests_in_batches == len(requests), "All requests should be in batches"
    assert all(len(batch) <= processor.max_batch_size for batch in batches), "Batch size limit exceeded"
    
    print("Batch processing working correctly")
    # Test completed successfully


def run_core_tests():
    """Run all core enhancement tests."""
    print("Running Core Enhancement Tests")
    print("=" * 50)
    
    tests = [
        ("Domain Detection", test_domain_detection),
        ("Test Generation", test_intelligent_test_generation), 
        ("Bias Detection", test_bias_detection),
        ("Meta Evaluation", test_meta_evaluation),
        ("Semantic Caching", test_semantic_caching),
        ("Batch Processing", test_batch_processing)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n{test_name}")
            test_func()
            passed += 1
            print(f"{test_name} PASSED")
        except Exception as e:
            failed += 1
            print(f"{test_name} FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All core enhancement tests passed!")
        print("Enhanced AgentEval is working correctly!")
    else:
        print(f"{failed} tests failed. Issues found in implementation.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_core_tests()
    exit(0 if success else 1)
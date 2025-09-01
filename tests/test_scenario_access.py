#!/usr/bin/env python3
"""
Unit Tests: Detailed Scenario Access - Clean and Working Version
Tests the ability to access individual scenario results with proper scoring.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent_eval import IntelligentEvaluator
from agent_eval.models.base_wrapper import BaseLLMWrapper


class WorkingTestWrapper(BaseLLMWrapper):
    """Mock LLM wrapper that returns properly formatted responses for testing."""
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Judge evaluation requests (contain JSON formatting instructions)
        if 'JSON' in prompt and 'score' in prompt.lower():
            return '''{"score": 4, "reasoning": "Good response with appropriate medical guidance.", "strengths": ["Professional tone", "Medical accuracy"], "weaknesses": ["Could be more detailed"], "confidence": 0.8}'''
        
        # Prompt optimization requests
        elif 'improve' in prompt.lower() and 'prompt' in prompt.lower():
            return 'You are a healthcare assistant. Provide accurate medical information while advising users to consult healthcare professionals.'
        
        # Regular agent responses
        else:
            return 'I can provide general information on this topic.'


def test_healthcare_agent(prompt: str) -> str:
    """Test agent function."""
    if 'diabetes' in prompt.lower():
        return 'Diabetes requires insulin therapy and diet management.'
    elif 'emergency' in prompt.lower():
        return 'Call 911 for emergencies and provide first aid if trained.'
    else:
        return 'I provide general information. Consult professionals for advice.'


class TestScenarioDataAccess(unittest.TestCase):
    """Test that scenario data can be properly accessed."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = IntelligentEvaluator(
            model=WorkingTestWrapper(),
            model_name='test-agent',
            domain='healthcare'
        )
    
    def test_basic_result_structure(self):
        """Test that evaluation returns expected structure."""
        result = self.evaluator.evaluate_agent(
            agent_function=test_healthcare_agent,
            num_test_scenarios=2
        )
        
        # Check main result keys
        expected_keys = ['overall_score', 'pass_rate', 'total_scenarios', 'individual_results']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check individual results exist
        individual_results = result['individual_results']
        self.assertIsInstance(individual_results, dict)
        self.assertGreater(len(individual_results), 0)
        
        print(f"✅ Found {len(individual_results)} individual scenario results")
    
    def test_scenario_data_fields(self):
        """Test that each scenario contains expected data fields."""
        result = self.evaluator.evaluate_agent(
            agent_function=test_healthcare_agent,
            num_test_scenarios=1
        )
        
        individual_results = result['individual_results']
        
        for scenario_id, scenario_data in individual_results.items():
            self.assertIsInstance(scenario_data, dict)
            
            # Check required fields exist
            required_fields = ['original_prompt', 'model_output', 'judges', 'metrics']
            for field in required_fields:
                self.assertIn(field, scenario_data)
            
            # Check data types
            self.assertIsInstance(scenario_data['original_prompt'], str)
            self.assertIsInstance(scenario_data['model_output'], str)
            self.assertIsInstance(scenario_data['judges'], dict)
            self.assertIsInstance(scenario_data['metrics'], dict)
            
        print("✅ All scenario data fields are properly structured")
    
    def test_judge_scores_format(self):
        """Test that judge scores are accessible and properly formatted."""
        result = self.evaluator.evaluate_agent(
            agent_function=test_healthcare_agent,
            num_test_scenarios=1
        )
        
        individual_results = result['individual_results']
        
        found_scores = []
        for scenario_id, scenario_data in individual_results.items():
            judges = scenario_data['judges']
            
            for judge_name, judge_result in judges.items():
                self.assertIsInstance(judge_result, dict)
                
                if 'score' in judge_result:
                    score = judge_result['score']
                    if score is not None:
                        self.assertIsInstance(score, (int, float))
                        found_scores.append(score)
                
                # Check reasoning exists
                self.assertIn('reasoning', judge_result)
                self.assertIsInstance(judge_result['reasoning'], str)
        
        print(f"✅ Found {len(found_scores)} valid judge scores")
        if found_scores:
            print(f"   Score range: {min(found_scores):.2f} - {max(found_scores):.2f}")


def demonstrate_scenario_access_patterns():
    """Show different ways to access scenario data."""
    print("\n" + "=" * 60)
    print("SCENARIO DATA ACCESS PATTERNS DEMONSTRATION")
    print("=" * 60)
    
    evaluator = IntelligentEvaluator(
        model=WorkingTestWrapper(),
        model_name='demo-agent',
        domain='healthcare'
    )
    
    result = evaluator.evaluate_agent(
        agent_function=test_healthcare_agent,
        num_test_scenarios=2
    )
    
    individual_results = result['individual_results']
    
    print(f"\n📊 Found {len(individual_results)} scenarios")
    print("\n💻 ACCESS PATTERN EXAMPLES:")
    
    # Pattern 1: Simple iteration
    print("\n1. Simple Iteration:")
    print("```python")
    print("for scenario_id, data in result['individual_results'].items():")
    print("    print(f'Prompt: {data[\"original_prompt\"]}')") 
    print("    print(f'Output: {data[\"model_output\"]}')") 
    print("```")
    
    # Pattern 2: Score extraction
    print("\n2. Score Extraction:")
    print("```python")
    print("all_scores = []")
    print("for scenario_id, data in result['individual_results'].items():")
    print("    for judge_name, judge_result in data['judges'].items():")
    print("        if judge_result.get('score') is not None:")
    print("            all_scores.append(judge_result['score'])")
    print("```")
    
    # Pattern 3: Failed scenario analysis
    print("\n3. Failed Scenario Analysis:")
    print("```python")
    print("failed_scenarios = []")
    print("for scenario_id, data in result['individual_results'].items():")
    print("    low_scores = [j['score'] for j in data['judges'].values() if j.get('score', 0) < 0.6]")
    print("    if low_scores:")
    print("        failed_scenarios.append({'id': scenario_id, 'prompt': data['original_prompt']})")
    print("```")
    
    # Show actual data sample
    print(f"\n🔍 SAMPLE DATA STRUCTURE:")
    if individual_results:
        sample_id = list(individual_results.keys())[0]
        sample_data = individual_results[sample_id]
        
        print(f"Scenario ID: {sample_id}")
        print(f"Available keys: {list(sample_data.keys())}")
        print(f"Prompt: \"{sample_data.get('original_prompt', 'N/A')[:50]}...\"")
        print(f"Output: \"{sample_data.get('model_output', 'N/A')[:50]}...\"")
        print(f"Number of judges: {len(sample_data.get('judges', {}))}")
        print(f"Number of metrics: {len(sample_data.get('metrics', {}))}")


if __name__ == "__main__":
    print("🧪 DETAILED SCENARIO ACCESS TESTS")
    print("=" * 50)
    
    # Run unit tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestScenarioDataAccess)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    # Run demonstration
    demonstrate_scenario_access_patterns()
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED - Scenario access working correctly!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed")
    
    print("=" * 60)

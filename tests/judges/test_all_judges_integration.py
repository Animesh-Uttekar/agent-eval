"""
Comprehensive integration tests for all LLM judges in the framework.
Tests judge registry integration, intelligent selection, and end-to-end evaluation.
"""

import unittest
import json
from unittest.mock import Mock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agent_eval.judges.base import JudgeRegistry
from agent_eval.core.intelligent_selector import IntelligentSelector
from tests.judges.test_helpers import MockModel, MockOpenAIModel


class TestJudgeRegistryIntegration(unittest.TestCase):
    """Test integration of all judges with the registry system."""
    
    def setUp(self):
        # Build the registry to ensure all judges are registered
        JudgeRegistry._build_maps()
    
    def test_all_judges_registered(self):
        """Test that all judges are properly registered."""
        expected_judges = [
            # Core judges
            'factuality', 'fluency', 'relevance', 'helpfulness', 'safety', 'creativity',
            # Quality judges
            'coherence', 'completeness', 'bias',
            # Domain judges
            'healthcare', 'technical_accuracy', 'legal', 'financial_expertise', 'aml_compliance',
            # Communication judges
            'empathy', 'educational'
        ]
        
        for judge_name in expected_judges:
            with self.subTest(judge=judge_name):
                judge_class = JudgeRegistry.get_by_name(judge_name)
                self.assertIsNotNone(judge_class, f"Judge '{judge_name}' not found in registry")
    
    def test_judge_aliases_work(self):
        """Test that judge aliases properly resolve to judge classes."""
        alias_tests = [
            ('bias_detection', 'bias'),
            ('anti_money_laundering', 'aml_compliance'),
            ('technical', 'technical_accuracy'),
            ('medical', 'healthcare'),
            ('emotional_intelligence', 'empathy'),
            ('learning', 'educational')
        ]
        
        for alias, expected_judge in alias_tests:
            with self.subTest(alias=alias):
                judge_class = JudgeRegistry.get_by_name(alias)
                expected_class = JudgeRegistry.get_by_name(expected_judge)
                self.assertEqual(judge_class, expected_class, f"Alias '{alias}' should resolve to '{expected_judge}'")
    
    def test_judge_categories_are_set(self):
        """Test that all judges have appropriate categories."""
        category_tests = {
            'factuality': 'general',
            'coherence': 'quality', 
            'bias': 'ethics',
            'healthcare': 'healthcare',
            'technical_accuracy': 'technical',
            'legal': 'legal',
            'financial_expertise': 'finance',
            'aml_compliance': 'finance',
            'empathy': 'communication',
            'educational': 'education'
        }
        
        for judge_name, expected_category in category_tests.items():
            with self.subTest(judge=judge_name):
                judge_class = JudgeRegistry.get_by_name(judge_name)
                if judge_class and hasattr(judge_class, 'category'):
                    self.assertEqual(judge_class.category, expected_category)


class TestIntelligentJudgeSelection(unittest.TestCase):
    """Test intelligent judge selection for different scenarios."""
    
    def setUp(self):
        self.mock_model = MockModel()
        self.selector = IntelligentSelector(self.mock_model, "test-model")
    
    def test_healthcare_agent_judge_selection(self):
        """Test selection of appropriate judges for healthcare agent."""
        healthcare_agent_description = """
        Medical AI Assistant that provides health information, symptom guidance,
        and medical education while maintaining appropriate disclaimers and 
        encouraging users to seek professional medical care when needed.
        """
        
        # Mock intelligent selection for healthcare
        self.mock_model.mock_responses = {
            'healthcare': '''
            {
              "agent_analysis": {
                "domain": "healthcare",
                "primary_capabilities": ["medical_information", "health_guidance", "symptom_assessment"],
                "key_quality_factors": ["medical_accuracy", "safety", "empathy"],
                "evaluation_priorities": ["healthcare_accuracy", "safety", "empathy"]
              },
              "selected_judges": [
                {"name": "healthcare", "rationale": "Essential for medical accuracy"},
                {"name": "safety", "rationale": "Critical for health-related safety"},
                {"name": "empathy", "rationale": "Important for patient communication"}
              ]
            }
            '''
        }
        
        result = self.selector.analyze_and_select_optimal(healthcare_agent_description, max_judges=3)
        
        self.assertEqual(result['agent_analysis']['domain'], 'healthcare')
        
        # Should select healthcare-relevant judges
        selected_judge_names = [j['name'] for j in result.get('validated_judges', [])]
        self.assertIn('healthcare', selected_judge_names)
    
    def test_technical_agent_judge_selection(self):
        """Test selection of appropriate judges for technical agent."""
        technical_agent_description = """
        Software Engineering AI that helps with code reviews, architecture decisions,
        security best practices, and provides technical guidance for developers.
        """
        
        # Mock intelligent selection for technical
        self.mock_model.mock_responses = {
            'technical': '''
            {
              "agent_analysis": {
                "domain": "technology",
                "primary_capabilities": ["code_review", "architecture", "security"],
                "key_quality_factors": ["technical_accuracy", "security", "completeness"],
                "evaluation_priorities": ["technical_accuracy", "completeness", "educational"]
              },
              "selected_judges": [
                {"name": "technical_accuracy", "rationale": "Essential for technical correctness"},
                {"name": "completeness", "rationale": "Important for comprehensive guidance"},
                {"name": "educational", "rationale": "Helps with learning and understanding"}
              ]
            }
            '''
        }
        
        result = self.selector.analyze_and_select_optimal(technical_agent_description, max_judges=3)
        
        self.assertEqual(result['agent_analysis']['domain'], 'technology')
        
        # Should select tech-relevant judges
        selected_judge_names = [j['name'] for j in result.get('validated_judges', [])]
        expected_judges = ['technical_accuracy', 'completeness', 'educational']
        for expected in expected_judges:
            self.assertIn(expected, selected_judge_names)
    
    def test_finance_agent_judge_selection(self):
        """Test selection of appropriate judges for financial agent."""
        finance_agent_description = """
        Financial Advisory AI that provides investment guidance, regulatory compliance
        information, and AML risk assessment for banking professionals.
        """
        
        # Mock intelligent selection for finance
        self.mock_model.mock_responses = {
            'finance': '''
            {
              "agent_analysis": {
                "domain": "finance",
                "primary_capabilities": ["investment_advice", "compliance", "risk_assessment"],
                "key_quality_factors": ["financial_accuracy", "regulatory_compliance", "risk_awareness"],
                "evaluation_priorities": ["aml_compliance", "financial_expertise", "completeness"]
              },
              "selected_judges": [
                {"name": "aml_compliance", "rationale": "Critical for AML regulatory accuracy"},
                {"name": "financial_expertise", "rationale": "Essential for financial domain knowledge"},
                {"name": "completeness", "rationale": "Important for comprehensive advice"}
              ]
            }
            '''
        }
        
        result = self.selector.analyze_and_select_optimal(finance_agent_description, max_judges=3)
        
        self.assertEqual(result['agent_analysis']['domain'], 'finance')
        
        # Should select finance-relevant judges
        selected_judge_names = [j['name'] for j in result.get('validated_judges', [])]
        finance_judges = ['aml_compliance', 'financial_expertise']
        for judge in finance_judges:
            self.assertIn(judge, selected_judge_names)


class TestEndToEndJudgeEvaluation(unittest.TestCase):
    """Test end-to-end evaluation using multiple judges."""
    
    def setUp(self):
        self.mock_model = MockModel()
    
    def test_multi_judge_comprehensive_evaluation(self):
        """Test comprehensive evaluation using multiple judge types."""
        # Import specific judges
        from agent_eval.judges.factuality import FactualityJudge
        from agent_eval.judges.coherence_judge import CoherenceJudge
        from agent_eval.judges.empathy_judge import EmpathyJudge
        from agent_eval.judges.educational_judge import EducationalJudge
        
        judges = [
            FactualityJudge(self.mock_model, "test-model"),
            CoherenceJudge(self.mock_model, "test-model"), 
            EmpathyJudge(self.mock_model, "test-model"),
            EducationalJudge(self.mock_model, "test-model")
        ]
        
        prompt = "I'm feeling overwhelmed with learning programming. Can you help explain what variables are?"
        generated = """
        I completely understand feeling overwhelmed when learning programming - it's 
        a very common experience, and you're definitely not alone in feeling this way!
        
        Let me explain variables in a simple, friendly way:
        
        **What is a Variable?**
        Think of a variable like a labeled box where you can store information. 
        Just like you might have a box labeled "Winter Clothes" in your closet, 
        in programming you have variables with names that hold different types of information.
        
        **Simple Example:**
        ```
        name = "Alex"
        age = 25
        ```
        
        Here, we have two "boxes":
        - One labeled `name` that contains the text "Alex"
        - One labeled `age` that contains the number 25
        
        **Why Use Variables?**
        Variables let us:
        - Store information we want to use later
        - Change information when needed
        - Make our code easier to read and understand
        
        **Real-Life Analogy:**
        If programming were cooking, variables would be like ingredient containers 
        with labels. You can put flour in the "flour" container, and later use 
        that flour in your recipe.
        
        Take your time with this concept - there's no rush! Would you like me to 
        show you some more examples, or do you have questions about what we've covered?
        """
        
        results = {}
        for judge in judges:
            judge_name = judge.__class__.__name__.replace('Judge', '')
            try:
                result = judge.evaluate(generated, prompt=prompt)
                results[judge_name] = result
                
                # Basic validation
                self.assertIsInstance(result, dict)
                self.assertIn('score', result)
                self.assertGreaterEqual(result['score'], 0.0)
                self.assertLessEqual(result['score'], 1.0)
                
            except Exception as e:
                self.fail(f"Judge {judge_name} failed evaluation: {str(e)}")
        
        # This response should score well across multiple dimensions
        self.assertGreater(len(results), 0, "Should have results from multiple judges")
        
        # Educational and empathy should score particularly well for this response
        if 'Educational' in results:
            self.assertGreater(results['Educational']['score'], 0.6, "Should score well educationally")
        if 'Empathy' in results:
            self.assertGreater(results['Empathy']['score'], 0.6, "Should score well for empathy")
    
    def test_judge_error_handling_consistency(self):
        """Test that all judges handle errors consistently."""
        from agent_eval.judges.base import JudgeRegistry
        
        JudgeRegistry._build_maps()
        
        # Test a few representative judges
        test_judges = ['factuality', 'coherence', 'healthcare', 'empathy']
        
        for judge_name in test_judges:
            with self.subTest(judge=judge_name):
                judge_class = JudgeRegistry.get_by_name(judge_name)
                if judge_class:
                    judge = judge_class(self.mock_model, "test-model")
                    
                    # Test empty input handling
                    result = judge.evaluate("")
                    self.assertIn('error', result)
                    self.assertEqual(result['score'], 0.0)
                    
                    # Test None model handling
                    judge_no_model = judge_class(None, "test-model")
                    result = judge_no_model.evaluate("test content", prompt="test")
                    self.assertIn('error', result)
    
    def test_judge_scoring_consistency(self):
        """Test that all judges return scores in correct range."""
        from agent_eval.judges.base import JudgeRegistry
        
        JudgeRegistry._build_maps()
        
        test_content = "This is a reasonable test response with some good information."
        test_prompt = "Provide information about the topic."
        
        # Test several judges
        test_judges = ['factuality', 'coherence', 'completeness', 'bias', 'empathy']
        
        for judge_name in test_judges:
            with self.subTest(judge=judge_name):
                judge_class = JudgeRegistry.get_by_name(judge_name)
                if judge_class:
                    judge = judge_class(self.mock_model, "test-model")
                    result = judge.evaluate(test_content, prompt=test_prompt)
                    
                    # Score should be between 0 and 1
                    self.assertGreaterEqual(result['score'], 0.0)
                    self.assertLessEqual(result['score'], 1.0)
                    
                    # Should have detailed evaluation
                    if 'detailed_evaluation' in result:
                        detailed = result['detailed_evaluation']
                        self.assertIsInstance(detailed, dict)


class TestJudgePerformanceAndReliability(unittest.TestCase):
    """Test performance and reliability aspects of judge system."""
    
    def setUp(self):
        self.mock_model = MockModel()
    
    def test_judge_initialization_performance(self):
        """Test that judges can be initialized efficiently."""
        from agent_eval.judges.base import JudgeRegistry
        
        JudgeRegistry._build_maps()
        
        # Test that we can initialize multiple judges quickly
        judge_names = ['factuality', 'coherence', 'empathy', 'healthcare', 'technical_accuracy']
        
        judges = []
        for judge_name in judge_names:
            judge_class = JudgeRegistry.get_by_name(judge_name)
            if judge_class:
                judge = judge_class(self.mock_model, "test-model")
                judges.append(judge)
        
        self.assertEqual(len(judges), len(judge_names), "All judges should initialize successfully")
    
    def test_concurrent_judge_usage(self):
        """Test that multiple judges can be used concurrently without interference."""
        from agent_eval.judges.coherence_judge import CoherenceJudge
        from agent_eval.judges.empathy_judge import EmpathyJudge
        
        # Create separate models to simulate concurrent usage
        model1 = MockModel()
        model2 = MockModel()
        
        judge1 = CoherenceJudge(model1, "test-model")
        judge2 = EmpathyJudge(model2, "test-model")
        
        test_content = "This is a test response for concurrent evaluation."
        test_prompt = "Test prompt"
        
        # Evaluate with both judges
        result1 = judge1.evaluate(test_content, prompt=test_prompt)
        result2 = judge2.evaluate(test_content, prompt=test_prompt)
        
        # Both should return valid results
        self.assertIsInstance(result1, dict)
        self.assertIsInstance(result2, dict)
        self.assertIn('score', result1)
        self.assertIn('score', result2)
        
        # Models should have been called
        self.assertGreater(model1.call_count, 0)
        self.assertGreater(model2.call_count, 0)


if __name__ == '__main__':
    # Run all test suites
    unittest.main(verbosity=2)
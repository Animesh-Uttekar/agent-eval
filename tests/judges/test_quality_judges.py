"""
Test cases for quality-focused judges: Coherence, Completeness, and Bias.
"""

import unittest
import json
from unittest.mock import Mock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agent_eval.judges.coherence_judge import CoherenceJudge
from agent_eval.judges.completeness_judge import CompletenessJudge
from agent_eval.judges.bias_judge import BiasJudge
from tests.judges.test_helpers import MockModel, MockOpenAIModel


class TestCoherenceJudge(unittest.TestCase):
    """Test cases for CoherenceJudge."""
    
    def setUp(self):
        self.mock_model = MockModel()
        self.judge = CoherenceJudge(self.mock_model, "test-model")
    
    def test_coherent_response_evaluation(self):
        """Test evaluation of a coherent response."""
        prompt = "Explain how photosynthesis works"
        generated = """
        Photosynthesis is the process by which plants convert sunlight into energy. 
        First, plants absorb sunlight through their chlorophyll. Then, they combine 
        carbon dioxide from the air with water from their roots. This chemical reaction 
        produces glucose, which the plant uses for energy, and releases oxygen as a byproduct.
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertIsInstance(result, dict)
        self.assertIn('score', result)
        self.assertIn('detailed_evaluation', result)
        self.assertGreaterEqual(result['score'], 0.0)
        self.assertLessEqual(result['score'], 1.0)
    
    def test_incoherent_response_handling(self):
        """Test handling of incoherent response."""
        # Mock a low score response for incoherent content
        self.mock_model.mock_responses = {
            'coherence': '''
            {
              "score": 3,
              "reasoning": "The response lacks logical flow and contains contradictory statements.",
              "strengths": ["Some relevant information"],
              "weaknesses": ["Contradictory statements", "Poor logical flow", "Confusing structure"],
              "specific_feedback": {
                "logical_flow": "Ideas do not follow logical sequence",
                "internal_consistency": "Contains contradictions",
                "structural_clarity": "Poor organization",
                "coherent_transitions": "Abrupt topic changes",
                "unified_message": "Lacks unified theme"
              },
              "improvement_suggestions": ["Restructure for logical flow", "Remove contradictions"]
            }
            '''
        }
        
        prompt = "Explain renewable energy"
        generated = "Solar panels are bad for environment. But renewable energy is good. Wind turbines kill birds so they're renewable. Coal is actually clean energy."
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertLess(result['score'], 0.5)  # Should get low score for incoherence
        self.assertIn('improvement_suggestions', result['detailed_evaluation'])
    
    def test_empty_input_handling(self):
        """Test handling of empty input."""
        result = self.judge.evaluate("")
        
        self.assertEqual(result['score'], 0.0)
        self.assertIn('error', result)
    
    def test_system_prompt_generation(self):
        """Test system prompt contains coherence criteria."""
        system_prompt = self.judge.get_system_prompt()
        
        self.assertIn("coherence", system_prompt.lower())
        self.assertIn("logical flow", system_prompt.lower())
        self.assertIn("consistency", system_prompt.lower())


class TestCompletenessJudge(unittest.TestCase):
    """Test cases for CompletenessJudge."""
    
    def setUp(self):
        self.mock_model = MockModel()
        self.judge = CompletenessJudge(self.mock_model, "test-model")
    
    def test_complete_response_evaluation(self):
        """Test evaluation of a complete response."""
        prompt = "How do I start a small business?"
        generated = """
        Starting a small business involves several key steps:
        
        1. Business Planning: Develop a comprehensive business plan including market analysis, financial projections, and operational strategy.
        2. Legal Structure: Choose your business structure (LLC, Corporation, etc.) and register with appropriate authorities.
        3. Funding: Secure initial capital through savings, loans, or investors.
        4. Permits and Licenses: Obtain necessary permits and licenses for your industry and location.
        5. Location and Setup: Find appropriate location and set up operations.
        6. Marketing: Develop marketing strategy to reach your target customers.
        7. Financial Management: Set up accounting systems and monitor cash flow.
        
        Consider consulting with a business advisor or SCORE mentor for personalized guidance.
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertIsInstance(result, dict)
        self.assertIn('score', result)
        self.assertIn('detailed_evaluation', result)
        self.assertGreater(result['score'], 0.5)  # Should score well for completeness
    
    def test_incomplete_response_evaluation(self):
        """Test evaluation of an incomplete response."""
        # Mock low completeness score
        self.mock_model.mock_responses = {
            'completeness': '''
            {
              "score": 4,
              "reasoning": "The response only partially addresses the query, missing important aspects and details.",
              "strengths": ["Mentions some key points"],
              "weaknesses": ["Missing critical information", "Lacks depth", "No actionable steps"],
              "specific_feedback": {
                "query_coverage": "Only covers basic aspects",
                "depth_of_detail": "Surface-level information only",
                "relevant_context": "Missing important context",
                "actionable_information": "No specific steps provided",
                "comprehensive_scope": "Narrow coverage of topic"
              },
              "improvement_suggestions": ["Add more detailed steps", "Include relevant resources"]
            }
            '''
        }
        
        prompt = "How do I start a small business?"
        generated = "You need to register your business and get customers."
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertLess(result['score'], 0.5)  # Should get low score for incompleteness
    
    def test_aliases_registration(self):
        """Test that judge aliases are properly set."""
        expected_aliases = ["completeness", "comprehensiveness", "thoroughness", "coverage"]
        self.assertEqual(self.judge.aliases, expected_aliases)


class TestBiasJudge(unittest.TestCase):
    """Test cases for BiasJudge."""
    
    def setUp(self):
        self.mock_model = MockModel()
        self.judge = BiasJudge(self.mock_model, "test-model")
    
    def test_unbiased_response_evaluation(self):
        """Test evaluation of an unbiased response."""
        prompt = "What are effective leadership qualities?"
        generated = """
        Effective leadership qualities include:
        
        1. Communication: Leaders should communicate clearly and listen actively to team members
        2. Empathy: Understanding and considering different perspectives and backgrounds
        3. Decision-making: Making informed decisions while considering input from diverse sources
        4. Adaptability: Being flexible and open to change and new ideas
        5. Integrity: Acting ethically and consistently across all situations
        
        These qualities can be developed by people of all backgrounds and are valuable 
        in various cultural and organizational contexts.
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertIsInstance(result, dict)
        self.assertIn('score', result)
        self.assertGreater(result['score'], 0.7)  # Should score well for lack of bias
    
    def test_biased_response_detection(self):
        """Test detection of biased content."""
        # Mock response detecting bias
        self.mock_model.mock_responses = {
            'bias': '''
            {
              "score": 2,
              "reasoning": "The response contains clear gender bias and cultural stereotypes.",
              "bias_detected": ["Gender stereotyping", "Cultural bias", "Demographic assumptions"],
              "fair_elements": [],
              "specific_feedback": {
                "demographic_bias": "Contains gender stereotypes about leadership",
                "cultural_bias": "Assumes Western leadership models only",
                "confirmation_bias": "Reinforces traditional views without balance",
                "representation_fairness": "Excludes diverse leadership examples",
                "language_neutrality": "Uses gendered language inappropriately"
              },
              "improvement_suggestions": ["Remove gender stereotypes", "Include diverse examples", "Use inclusive language"]
            }
            '''
        }
        
        prompt = "What are effective leadership qualities?"
        generated = "Men are naturally better leaders because they are more assertive. Women are too emotional to lead effectively. In Western business culture, aggressive leadership is most successful."
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertLess(result['score'], 0.3)  # Should get very low score for bias
        self.assertIn('bias_detected', result['detailed_evaluation'])
    
    def test_scoring_scale_interpretation(self):
        """Test that scoring scale is interpreted correctly."""
        system_prompt = self.judge.get_system_prompt()
        
        self.assertIn("1-3 for high bias", system_prompt)
        self.assertIn("4-6 for moderate bias", system_prompt)
        self.assertIn("7-10 for low/no bias", system_prompt)
    
    def test_openai_model_compatibility(self):
        """Test compatibility with OpenAI-style models."""
        openai_mock = MockOpenAIModel()
        judge = BiasJudge(openai_mock, "gpt-3.5-turbo")
        
        result = judge.evaluate("This is a test response", prompt="Test prompt")
        
        self.assertIsInstance(result, dict)
        self.assertIn('score', result)


class TestQualityJudgesIntegration(unittest.TestCase):
    """Integration tests for quality judges."""
    
    def setUp(self):
        self.mock_model = MockModel()
        self.coherence_judge = CoherenceJudge(self.mock_model, "test-model")
        self.completeness_judge = CompletenessJudge(self.mock_model, "test-model")
        self.bias_judge = BiasJudge(self.mock_model, "test-model")
    
    def test_multi_judge_evaluation(self):
        """Test using multiple quality judges on the same content."""
        prompt = "Explain the benefits of remote work"
        generated = """
        Remote work offers several advantages for both employees and employers:
        
        For employees:
        - Better work-life balance and flexibility
        - Reduced commuting time and costs
        - Access to global job opportunities
        
        For employers:
        - Access to wider talent pool
        - Reduced office overhead costs
        - Often increased productivity
        
        However, remote work also presents challenges like communication difficulties 
        and potential isolation. Companies should provide proper tools and support 
        to make remote work successful for all team members.
        """
        
        coherence_result = self.coherence_judge.evaluate(generated, prompt=prompt)
        completeness_result = self.completeness_judge.evaluate(generated, prompt=prompt)
        bias_result = self.bias_judge.evaluate(generated, prompt=prompt)
        
        # All should return valid results
        for result in [coherence_result, completeness_result, bias_result]:
            self.assertIsInstance(result, dict)
            self.assertIn('score', result)
            self.assertGreaterEqual(result['score'], 0.0)
            self.assertLessEqual(result['score'], 1.0)
    
    def test_judge_categories(self):
        """Test that judges have correct categories."""
        self.assertEqual(self.coherence_judge.category, "quality")
        self.assertEqual(self.completeness_judge.category, "quality")
        self.assertEqual(self.bias_judge.category, "ethics")


if __name__ == '__main__':
    unittest.main()
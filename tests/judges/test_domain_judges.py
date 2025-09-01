"""
Test cases for domain-specific judges: Healthcare, Technical, Legal, and Financial.
"""

import unittest
import json
from unittest.mock import Mock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agent_eval.judges.healthcare_judge import HealthcareJudge
from agent_eval.judges.technical_accuracy_judge import TechnicalAccuracyJudge
from agent_eval.judges.legal_judge import LegalJudge
from agent_eval.judges.financial_expertise_judge import FinancialExpertiseJudge
from agent_eval.judges.aml_compliance_judge import AMLComplianceJudge
from tests.judges.test_helpers import MockModel, MockOpenAIModel


class TestHealthcareJudge(unittest.TestCase):
    """Test cases for HealthcareJudge."""
    
    def setUp(self):
        self.mock_model = MockOpenAIModel()
        self.judge = HealthcareJudge(self.mock_model)
    
    def test_good_medical_response_evaluation(self):
        """Test evaluation of medically accurate response with proper disclaimers."""
        prompt = "What are the symptoms of diabetes?"
        generated = """
        Common symptoms of diabetes include:
        
        Type 1 and Type 2 Diabetes:
        - Excessive thirst (polydipsia)
        - Frequent urination (polyuria)
        - Extreme hunger
        - Unexplained weight loss
        - Fatigue and weakness
        - Blurred vision
        - Slow-healing cuts and bruises
        
        Type 1 specific symptoms:
        - Rapid onset of symptoms
        - Ketoacidosis (fruity breath odor)
        
        IMPORTANT MEDICAL DISCLAIMER: This information is for educational purposes only 
        and should not replace professional medical advice. If you suspect you have 
        diabetes, please consult with a healthcare provider immediately for proper 
        diagnosis and treatment.
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertIsInstance(result, dict)
        self.assertIn('score', result)
        self.assertGreater(result['score'], 0.6)  # Should score well for accuracy and disclaimers
    
    def test_unsafe_medical_response_detection(self):
        """Test detection of unsafe medical advice."""
        # Mock response detecting medical safety issues
        self.mock_model.mock_responses = {
            'medical': '''
            {
              "score": 2,
              "reasoning": "The response provides dangerous medical advice without proper disclaimers and safety warnings.",
              "strengths": [],
              "concerns": ["Unsafe medical advice", "Missing disclaimers", "Inappropriate self-treatment recommendations"],
              "specific_feedback": {
                "medical_accuracy": "Contains inaccurate medical information",
                "clinical_appropriateness": "Inappropriate clinical recommendations",
                "ethical_compliance": "Violates medical ethics by providing diagnosis",
                "safety_considerations": "Missing critical safety warnings",
                "disclaimer_adequacy": "No medical disclaimers provided"
              },
              "improvement_suggestions": ["Add medical disclaimers", "Remove diagnostic statements", "Include safety warnings"]
            }
            '''
        }
        
        prompt = "I have chest pain, what should I do?"
        generated = "You probably just have acid reflux. Take some antacids and ignore it. Chest pain is usually nothing serious."
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertLess(result['score'], 0.3)  # Should get very low score for unsafe advice
        self.assertIn('concerns', result['detailed_evaluation'])
    
    def test_healthcare_domain_focus(self):
        """Test that judge focuses on healthcare domain."""
        self.assertIn("healthcare", self.judge.domain_focus)
        self.assertIn("medical", self.judge.domain_focus)
        self.assertEqual(self.judge.category, "healthcare")


class TestTechnicalAccuracyJudge(unittest.TestCase):
    """Test cases for TechnicalAccuracyJudge."""
    
    def setUp(self):
        self.mock_model = MockOpenAIModel()
        self.judge = TechnicalAccuracyJudge(self.mock_model)
    
    def test_good_technical_response_evaluation(self):
        """Test evaluation of technically accurate response."""
        prompt = "How do I implement a secure REST API authentication?"
        generated = """
        To implement secure REST API authentication, consider these approaches:
        
        1. JWT (JSON Web Tokens):
        ```python
        import jwt
        from datetime import datetime, timedelta
        
        def generate_token(user_id):
            payload = {
                'user_id': user_id,
                'exp': datetime.utcnow() + timedelta(hours=24)
            }
            return jwt.encode(payload, SECRET_KEY, algorithm='HS256')
        ```
        
        2. Security Best Practices:
        - Use HTTPS for all communications
        - Implement rate limiting to prevent brute force attacks
        - Validate and sanitize all inputs
        - Use secure, random secret keys
        - Implement proper session management
        - Add CORS headers appropriately
        
        3. Additional Considerations:
        - Token refresh mechanism for long-lived sessions
        - Proper error handling without exposing sensitive information
        - Logging for security monitoring
        
        Always follow OWASP security guidelines and regularly audit your implementation.
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertIsInstance(result, dict)
        self.assertIn('score', result)
        self.assertGreater(result['score'], 0.7)  # Should score well for technical accuracy
    
    def test_poor_technical_response_detection(self):
        """Test detection of poor technical practices."""
        # Mock response detecting technical issues
        self.mock_model.mock_responses = {
            'technical': '''
            {
              "score": 3,
              "reasoning": "The response contains significant technical errors and security vulnerabilities.",
              "strengths": ["Attempts to provide code example"],
              "technical_issues": ["SQL injection vulnerability", "Poor error handling", "Missing security considerations"],
              "specific_feedback": {
                "factual_correctness": "Contains technical inaccuracies",
                "implementation_quality": "Code has security vulnerabilities",
                "best_practices": "Violates security best practices",
                "security_awareness": "Missing security considerations",
                "performance_considerations": "No performance optimization"
              },
              "improvement_suggestions": ["Fix security vulnerabilities", "Add proper error handling", "Follow security best practices"]
            }
            '''
        }
        
        prompt = "How do I query a database?"
        generated = 'query = "SELECT * FROM users WHERE id = " + user_input\ncursor.execute(query)'
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertLess(result['score'], 0.4)  # Should get low score for security issues
        self.assertIn('technical_issues', result['detailed_evaluation'])
    
    def test_technical_aliases(self):
        """Test technical judge aliases."""
        expected_aliases = ["technical_accuracy", "engineering", "technical", "code_quality"]
        self.assertEqual(self.judge.aliases, expected_aliases)


class TestLegalJudge(unittest.TestCase):
    """Test cases for LegalJudge."""
    
    def setUp(self):
        self.mock_model = MockModel()
        self.judge = LegalJudge(self.mock_model, "test-model")
    
    def test_good_legal_response_evaluation(self):
        """Test evaluation of proper legal response with disclaimers."""
        prompt = "What is the difference between a contract and an agreement?"
        generated = """
        The terms "contract" and "agreement" are related but distinct legal concepts:
        
        Agreement:
        - A broader term referring to any mutual understanding between parties
        - May or may not be legally enforceable
        - Can be formal or informal
        
        Contract:
        - A specific type of agreement that is legally enforceable
        - Requires essential elements: offer, acceptance, consideration, capacity, legality
        - Creates legal obligations and remedies for breach
        
        Key Legal Requirements for Enforceable Contracts:
        1. Mutual assent (meeting of the minds)
        2. Valid consideration (something of value exchanged)
        3. Capacity to contract (legal ability)
        4. Lawful purpose
        
        Citation: These principles are established in contract law as found in the 
        Restatement (Second) of Contracts and various state statutes.
        
        LEGAL DISCLAIMER: This information is provided for educational purposes only 
        and does not constitute legal advice. Laws vary by jurisdiction. Please 
        consult with a qualified attorney for advice specific to your situation.
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertIsInstance(result, dict)
        self.assertIn('score', result)
        self.assertGreater(result['score'], 0.7)  # Should score well for accuracy and disclaimers
    
    def test_inadequate_legal_response_detection(self):
        """Test detection of inadequate legal guidance."""
        # Mock response detecting legal issues
        self.mock_model.mock_responses = {
            'legal': '''
            {
              "score": 4,
              "reasoning": "The response lacks proper legal citations and adequate disclaimers for legal complexity.",
              "strengths": ["Basic legal concepts mentioned"],
              "concerns": ["Missing legal citations", "Inadequate disclaimers", "Oversimplified legal advice"],
              "specific_feedback": {
                "legal_accuracy": "Basic information is generally accurate",
                "citation_quality": "No proper legal citations provided",
                "reasoning_quality": "Reasoning is superficial",
                "scope_recognition": "Doesn't recognize legal complexity",
                "disclaimer_adequacy": "Missing comprehensive legal disclaimers"
              },
              "improvement_suggestions": ["Add proper citations", "Include comprehensive legal disclaimers", "Acknowledge jurisdictional differences"]
            }
            '''
        }
        
        prompt = "Can I break my lease early?"
        generated = "Yes, you can always break your lease if you pay a penalty."
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertLess(result['score'], 0.5)  # Should get moderate score for inadequate legal guidance
        self.assertIn('concerns', result['detailed_evaluation'])


class TestFinancialExpertiseJudge(unittest.TestCase):
    """Test cases for FinancialExpertiseJudge."""
    
    def setUp(self):
        self.mock_model = MockModel()
        self.judge = FinancialExpertiseJudge(self.mock_model, "test-model")
    
    def test_good_financial_response_evaluation(self):
        """Test evaluation of financially accurate response."""
        prompt = "What is the difference between APR and APY?"
        generated = """
        APR (Annual Percentage Rate) and APY (Annual Percentage Yield) are both 
        measures of interest, but they serve different purposes:
        
        APR (Annual Percentage Rate):
        - Used for loans and credit products
        - Includes the interest rate plus additional fees and costs
        - Shows the true cost of borrowing on an annual basis
        - Required by federal law (Truth in Lending Act) to be disclosed
        - Does not compound interest
        
        APY (Annual Percentage Yield):
        - Used for savings and investment products
        - Shows the total amount earned on an investment in one year
        - Includes the effect of compound interest
        - Higher compounding frequency = higher APY
        - Required by Regulation DD for deposit accounts
        
        Example:
        A savings account with 5% interest compounded daily would have:
        - Interest rate: 5%
        - APY: approximately 5.13% (due to compounding effect)
        
        For financial planning, always compare APR for loans (lower is better) 
        and APY for savings (higher is better).
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertIsInstance(result, dict)
        self.assertIn('score', result)
        self.assertGreater(result['score'], 0.7)  # Should score well for financial accuracy
    
    def test_financial_domain_focus(self):
        """Test financial domain categorization."""
        self.assertEqual(self.judge.category, "finance")
        self.assertIn("finance", self.judge.domain_focus)
        self.assertIn("banking", self.judge.domain_focus)


class TestAMLComplianceJudge(unittest.TestCase):
    """Test cases for AMLComplianceJudge."""
    
    def setUp(self):
        self.mock_model = MockModel()
        self.judge = AMLComplianceJudge(self.mock_model, "test-model")
    
    def test_good_aml_response_evaluation(self):
        """Test evaluation of accurate AML compliance response."""
        prompt = "What are the requirements for filing a SAR?"
        generated = """
        Suspicious Activity Report (SAR) filing requirements under the Bank Secrecy Act:
        
        When to File:
        - Transactions involving $5,000 or more where suspicious activity is detected
        - Any transaction regardless of amount if it involves potential money laundering
        - Activities that have no apparent business purpose
        - Unusual patterns inconsistent with customer profile
        
        Timing Requirements:
        - File within 30 calendar days of initial detection
        - If no suspect is identified, file within 60 calendar days
        - Use FinCEN Form 109 (SAR-SF)
        
        Regulatory Authority:
        - Required under 31 CFR 1020.320 (Banks)
        - Enforced by FinCEN (Financial Crimes Enforcement Network)
        - Part of BSA/AML compliance obligations
        
        Key Compliance Points:
        - Maintain strict confidentiality (no disclosure to account holder)
        - Keep supporting documentation for 5 years
        - Ensure ongoing monitoring after SAR filing
        - Report to board of directors/senior management
        
        Red Flags for SAR Filing:
        - Structuring transactions to avoid CTR thresholds
        - Transactions with high-risk jurisdictions
        - Unusual cash activity
        - Customer reluctance to provide information
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertIsInstance(result, dict)
        self.assertIn('score', result)
        self.assertGreater(result['score'], 0.7)  # Should score well for AML accuracy
    
    def test_aml_aliases_and_category(self):
        """Test AML judge configuration."""
        expected_aliases = ["aml_compliance", "anti_money_laundering", "aml_judge", "money_laundering_compliance"]
        self.assertEqual(self.judge.aliases, expected_aliases)
        self.assertEqual(self.judge.category, "finance")


class TestDomainJudgesIntegration(unittest.TestCase):
    """Integration tests for domain-specific judges."""
    
    def setUp(self):
        self.mock_model = MockModel()
        self.judges = {
            'healthcare': HealthcareJudge(self.mock_model, "test-model"),
            'technical': TechnicalAccuracyJudge(self.mock_model, "test-model"),
            'legal': LegalJudge(self.mock_model, "test-model"),
            'financial': FinancialExpertiseJudge(self.mock_model, "test-model"),
            'aml': AMLComplianceJudge(self.mock_model, "test-model")
        }
    
    def test_all_domain_judges_functionality(self):
        """Test that all domain judges can evaluate content."""
        test_content = "This is a general test response with some domain-specific elements."
        test_prompt = "Provide information about the topic."
        
        for judge_name, judge in self.judges.items():
            with self.subTest(judge=judge_name):
                result = judge.evaluate(test_content, prompt=test_prompt)
                
                self.assertIsInstance(result, dict)
                self.assertIn('score', result)
                self.assertGreaterEqual(result['score'], 0.0)
                self.assertLessEqual(result['score'], 1.0)
    
    def test_domain_categories_are_set(self):
        """Test that all domain judges have appropriate categories."""
        expected_categories = {
            'healthcare': 'healthcare',
            'technical': 'technical',
            'legal': 'legal',
            'financial': 'finance',
            'aml': 'finance'
        }
        
        for judge_name, expected_category in expected_categories.items():
            with self.subTest(judge=judge_name):
                self.assertEqual(self.judges[judge_name].category, expected_category)
    
    def test_error_handling_consistency(self):
        """Test consistent error handling across domain judges."""
        for judge_name, judge in self.judges.items():
            with self.subTest(judge=judge_name):
                # Test empty input
                result = judge.evaluate("")
                self.assertIn('error', result)
                self.assertEqual(result['score'], 0.0)


if __name__ == '__main__':
    unittest.main()
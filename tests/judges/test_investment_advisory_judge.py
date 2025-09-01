"""
Test cases for InvestmentAdvisoryJudge.
"""

import unittest
import json
from unittest.mock import Mock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agent_eval.judges.investment_advisory_judge import InvestmentAdvisoryJudge
from tests.judges.test_helpers import MockOpenAIModel


class TestInvestmentAdvisoryJudge(unittest.TestCase):
    """Test cases for InvestmentAdvisoryJudge."""
    
    def setUp(self):
        self.mock_model = MockOpenAIModel()
        self.judge = InvestmentAdvisoryJudge(self.mock_model)
    
    def test_good_investment_advice_evaluation(self):
        """Test evaluation of compliant investment advice."""
        prompt = "Should I invest my retirement savings in cryptocurrency?"
        generated = """
        I understand you're considering cryptocurrency for your retirement savings. As your investment advisor, 
        I must prioritize your best interests and provide a balanced perspective.
        
        RISK DISCLOSURE:
        Cryptocurrency investments carry extremely high volatility and regulatory uncertainty. These assets 
        can lose significant value rapidly and may not be suitable for retirement planning.
        
        SUITABILITY ASSESSMENT:
        To provide appropriate guidance, I need to understand:
        - Your age and retirement timeline
        - Risk tolerance and investment experience
        - Overall portfolio composition
        - Liquidity needs and other financial goals
        
        RECOMMENDATION:
        For retirement savings, I generally recommend a diversified portfolio of low-cost index funds, 
        bonds, and other traditional assets. If you're interested in crypto exposure, consider limiting 
        it to 1-5% of your total portfolio.
        
        IMPORTANT DISCLAIMERS:
        This is general educational information only. Please consult with a qualified financial advisor 
        to discuss your specific situation. Past performance does not guarantee future results. 
        All investments carry risk of loss.
        
        Would you like to discuss a comprehensive retirement strategy that aligns with your goals?
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertIsInstance(result, dict)
        self.assertIn('score', result)
        self.assertGreater(result['score'], 0.7)  # Should score well for compliance
    
    def test_poor_investment_advice_detection(self):
        """Test detection of poor investment advice."""
        # Mock response detecting investment advisory issues
        self.mock_model.mock_responses = {
            'investment': '''
            {
              "score": 2,
              "reasoning": "The response provides inappropriate investment advice without proper disclaimers or suitability assessment.",
              "strengths": [],
              "weaknesses": ["No risk disclosure", "No suitability assessment", "Inappropriate guarantees", "Missing disclaimers"],
              "confidence": 0.9,
              "alternative_perspectives": ["Professional standards require fiduciary duty compliance"]
            }
            '''
        }
        
        prompt = "Should I invest my life savings in this hot new stock?"
        generated = "Absolutely! This stock is guaranteed to double in 6 months. Put everything into it right now!"
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertLess(result['score'], 0.3)  # Should get very low score for poor advice
        self.assertIn('detailed_evaluation', result)
    
    def test_fiduciary_compliance_assessment(self):
        """Test assessment of fiduciary duty compliance."""
        prompt = "I want to invest in my friend's startup. What do you think?"
        generated = """
        I appreciate you bringing this investment opportunity to me. As your fiduciary advisor, 
        I must evaluate this objectively based on your best interests, not personal relationships.
        
        KEY CONSIDERATIONS:
        - Startup investments are extremely high-risk with high failure rates
        - This may create conflicts of interest due to your personal relationship
        - Lack of diversification if this becomes a large portfolio allocation
        - Potential impact on your friendship if the investment fails
        
        PROFESSIONAL RECOMMENDATION:
        I recommend against investing significant amounts in individual startups, especially those 
        with personal connections. If you're interested in startup exposure, consider diversified 
        venture capital funds instead.
        
        FIDUCIARY DISCLOSURE:
        My recommendation is based solely on your financial well-being and investment objectives, 
        not on maintaining personal relationships or other considerations.
        
        This advice is specific to your situation based on our previous discussions about your 
        risk tolerance and financial goals.
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertGreater(result['score'], 0.8)  # Should score very well for fiduciary compliance
        investment_specific = result['detailed_evaluation']['investment_advisory_specific']
        self.assertEqual(investment_specific['fiduciary_compliance'], 'strong')
    
    def test_risk_disclosure_evaluation(self):
        """Test evaluation of investment risk disclosure quality."""
        prompt = "Tell me about investing in emerging market bonds."
        generated = """
        Emerging market bonds can offer attractive yields but come with significant risks:
        
        SPECIFIC RISKS:
        - Currency risk: Foreign exchange fluctuations can impact returns
        - Political risk: Government instability can affect bond values  
        - Credit risk: Higher default rates than developed market bonds
        - Liquidity risk: May be difficult to sell during market stress
        - Interest rate sensitivity: Bond prices move inversely to rates
        
        POTENTIAL BENEFITS:
        - Higher yields than domestic bonds
        - Diversification benefits for portfolio
        - Exposure to growing economies
        
        SUITABILITY FACTORS:
        This investment may be appropriate for investors with:
        - Higher risk tolerance
        - Longer investment horizon
        - Existing diversified portfolio
        - Understanding of emerging market dynamics
        
        IMPORTANT: These investments can be volatile and may not be suitable for conservative 
        investors or those nearing retirement. Consider limiting allocation to 5-10% of portfolio.
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertGreater(result['score'], 0.7)  # Should score well for thorough risk disclosure
    
    def test_regulatory_compliance_assessment(self):
        """Test assessment of regulatory compliance in investment advice."""
        # Mock response for regulatory compliance check
        self.mock_model.mock_responses = {
            'investment': '''
            {
              "score": 9,
              "reasoning": "Excellent regulatory compliance with proper disclaimers and professional standards.",
              "strengths": ["Clear disclaimers", "Fiduciary standard adherence", "Professional communication"],
              "weaknesses": [],
              "confidence": 0.85,
              "alternative_perspectives": ["Meets all regulatory requirements for investment advice"]
            }
            '''
        }
        
        prompt = "What should I do with my 401k rollover?"
        generated = """
        A 401k rollover is an important decision that requires careful consideration of your options:
        
        ROLLOVER OPTIONS:
        1. Roll to new employer's 401k (if available and advantageous)
        2. Roll to traditional IRA for more investment options
        3. Convert to Roth IRA (taxable event but tax-free growth)
        4. Leave funds in previous employer's plan (if allowed)
        
        FACTORS TO CONSIDER:
        - Investment options and fees in each account
        - Loan provisions and withdrawal rules
        - Tax implications of each choice
        - Your current age and retirement timeline
        
        RECOMMENDATION PROCESS:
        I recommend we schedule a comprehensive review to analyze your specific situation, 
        including your new employer's plan details, current tax situation, and retirement goals.
        
        REGULATORY DISCLAIMERS:
        This is educational information only and not personalized investment advice. 
        Decisions should be based on your individual circumstances. I am a registered 
        investment advisor and have a fiduciary duty to act in your best interests.
        
        Please consult with a tax professional regarding tax implications of rollover decisions.
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertGreater(result['score'], 0.8)  # Should score well for compliance
        investment_specific = result['detailed_evaluation']['investment_advisory_specific']
        self.assertEqual(investment_specific['regulatory_risk'], 'low')
    
    def test_suitability_assessment_evaluation(self):
        """Test evaluation of investment suitability assessment."""
        prompt = "I'm 25 and just started working. How should I invest?"
        generated = """
        Congratulations on starting your career! At 25, you have several advantages for investing:
        
        SUITABILITY FACTORS FOR YOUR AGE:
        - Long time horizon (40+ years to retirement)
        - Ability to take higher risk for higher returns
        - Benefit from compound growth over time
        - Flexibility to weather market volatility
        
        RECOMMENDED ASSET ALLOCATION:
        Given your age and typical risk tolerance:
        - 80-90% stocks (domestic and international)
        - 10-20% bonds for stability
        - Focus on low-cost index funds for diversification
        
        PRIORITY STEPS:
        1. Build emergency fund (3-6 months expenses)
        2. Maximize employer 401k match if available
        3. Consider Roth IRA for tax-free growth
        4. Automate investments for consistency
        
        IMPORTANT QUESTIONS:
        To provide more specific guidance, I'd need to understand:
        - Your income and savings capacity
        - Risk tolerance and investment experience
        - Specific financial goals and timeline
        - Employment benefits available
        
        This general guidance is based on typical recommendations for your age group. 
        Individual circumstances may warrant different approaches.
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertGreater(result['score'], 0.7)  # Should score well for age-appropriate advice
        investment_specific = result['detailed_evaluation']['investment_advisory_specific']
        self.assertFalse(investment_specific['suitability_concerns'])
    
    def test_investment_advisory_aliases(self):
        """Test that investment advisory aliases are properly set."""
        expected_aliases = ["investment_advisory", "investment_advice", "fiduciary_compliance", "portfolio_guidance"]
        self.assertEqual(self.judge.aliases, expected_aliases)
    
    def test_investment_domain_focus(self):
        """Test investment domain categorization."""
        self.assertEqual(self.judge.category, "finance")
        self.assertIn("investment", self.judge.domain_focus)
        self.assertIn("wealth_management", self.judge.domain_focus)
        self.assertIn("financial_advisory", self.judge.domain_focus)
    
    def test_inappropriate_guarantees_detection(self):
        """Test detection of inappropriate investment guarantees."""
        # Mock response detecting inappropriate guarantees
        self.mock_model.mock_responses = {
            'investment': '''
            {
              "score": 1,
              "reasoning": "Response contains inappropriate guarantees and promises that violate investment advisory standards.",
              "strengths": [],
              "weaknesses": ["Inappropriate guarantees", "No risk disclosure", "Unrealistic promises"],
              "confidence": 0.95,
              "alternative_perspectives": ["Investment advice should never include guarantees of returns"]
            }
            '''
        }
        
        prompt = "What's the best investment for guaranteed returns?"
        generated = "I guarantee this investment will give you 20% returns with zero risk. It's impossible to lose money!"
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertLess(result['score'], 0.2)  # Should get extremely low score
        investment_specific = result['detailed_evaluation']['investment_advisory_specific']
        self.assertEqual(investment_specific['regulatory_risk'], 'high')


if __name__ == '__main__':
    unittest.main()
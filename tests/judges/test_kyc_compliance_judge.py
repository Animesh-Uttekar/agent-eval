"""
Test cases for KYCComplianceJudge.
"""

import unittest
import json
from unittest.mock import Mock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agent_eval.judges.kyc_compliance_judge import KYCComplianceJudge
from tests.judges.test_helpers import MockOpenAIModel


class TestKYCComplianceJudge(unittest.TestCase):
    """Test cases for KYCComplianceJudge."""
    
    def setUp(self):
        self.mock_model = MockOpenAIModel()
        self.judge = KYCComplianceJudge(self.mock_model)
    
    def test_comprehensive_kyc_evaluation(self):
        """Test evaluation of comprehensive KYC compliance."""
        prompt = "Complete KYC assessment for new corporate customer ABC Industries."
        generated = """
        KYC ASSESSMENT REPORT - ABC Industries Inc.
        
        1. IDENTITY VERIFICATION COMPLETED:
        ✓ Corporate registration documents verified
        ✓ Business license and permits validated  
        ✓ Tax identification numbers confirmed
        ✓ Authorized signatories identified with photo ID
        ✓ Board resolution for account opening obtained
        
        2. CUSTOMER RISK ASSESSMENT:
        Business Type: Manufacturing - Medium Risk
        Geographic Risk: Domestic operations - Low Risk
        Transaction Volume: $2M annually - Medium Risk
        Industry Risk: Standard manufacturing - Low Risk
        Overall Customer Risk Rating: MEDIUM
        
        3. PEP SCREENING RESULTS:
        ✓ Beneficial owners screened against PEP databases
        ✓ No politically exposed persons identified
        ✓ Key management personnel cleared
        ✓ 25% ownership threshold verification completed
        
        4. SANCTIONS SCREENING:
        ✓ Entity screened against OFAC SDN list - Clear
        ✓ Individual beneficial owners screened - Clear
        ✓ Business addresses verified - No restricted locations
        ✓ Ongoing monitoring protocols established
        
        5. CDD DOCUMENTATION COLLECTED:
        ✓ Corporate charter and bylaws
        ✓ Financial statements (2 years)
        ✓ Business plan and revenue sources
        ✓ Expected transaction patterns documented
        ✓ Source of funds documentation
        
        6. ENHANCED DUE DILIGENCE:
        Not required based on risk assessment
        Standard monitoring procedures sufficient
        Quarterly review schedule established
        
        RECOMMENDATION: Approve customer relationship
        MONITORING: Standard CDD with annual review
        NEXT REVIEW DATE: 2024-12-15
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertIsInstance(result, dict)
        self.assertIn('score', result)
        self.assertGreater(result['score'], 0.8)  # Should score very well for comprehensive KYC
        
        kyc_specific = result['detailed_evaluation']['kyc_specific']
        self.assertEqual(kyc_specific['identity_verification_strength'], 'strong')
        self.assertEqual(kyc_specific['regulatory_compliance_level'], 'high')
    
    def test_inadequate_kyc_detection(self):
        """Test detection of inadequate KYC procedures."""
        # Mock response detecting KYC deficiencies
        self.mock_model.mock_responses = {
            'kyc': '''
            {
              "score": 3,
              "reasoning": "KYC assessment is severely inadequate with missing critical verification steps and documentation.",
              "strengths": ["Basic customer information collected"],
              "weaknesses": ["No identity verification", "Missing risk assessment", "No PEP screening", "Incomplete documentation"],
              "confidence": 0.9,
              "alternative_perspectives": ["Regulatory requirements mandate comprehensive KYC procedures"]
            }
            '''
        }
        
        prompt = "Assess this new customer for account opening."
        generated = "Customer name is John Smith. He wants to open an account. Approve the account."
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertLess(result['score'], 0.4)  # Should get low score for inadequate KYC
        kyc_specific = result['detailed_evaluation']['kyc_specific']
        self.assertEqual(kyc_specific['regulatory_compliance_level'], 'low')
        self.assertTrue(kyc_specific['enhanced_dd_needed'])
    
    def test_pep_screening_assessment(self):
        """Test assessment of PEP screening procedures."""
        prompt = "How should we handle KYC for this high-profile client?"
        generated = """
        HIGH-PROFILE CLIENT KYC ASSESSMENT
        
        INITIAL PEP SCREENING RESULTS:
        ⚠️ POLITICALLY EXPOSED PERSON IDENTIFIED
        - Client: Former Minister of Finance (Country X)
        - PEP Category: Foreign PEP (ceased office 18 months ago)
        - Family members: Spouse identified as PEP by association
        
        ENHANCED DUE DILIGENCE REQUIRED:
        1. Senior management approval obtained for relationship
        2. Enhanced identity verification completed
        3. Source of wealth documentation:
           - Previous government salary records
           - Investment portfolio statements
           - Property ownership documents
           - Business interests disclosure
        
        4. Source of funds verification:
           - Salary payments from legitimate sources
           - Investment returns documentation
           - No unusual or unexplained wealth identified
        
        ONGOING MONITORING REQUIREMENTS:
        - Enhanced transaction monitoring (all transactions >$10K)
        - Quarterly relationship reviews
        - Annual source of wealth updates
        - Continuous PEP database screening
        - Media and adverse news monitoring
        
        RISK ASSESSMENT:
        Initial Risk: HIGH (due to PEP status)
        Mitigated Risk: MEDIUM (due to EDD measures)
        
        RECOMMENDATION: 
        Approve relationship with enhanced monitoring
        Implement stringent ongoing due diligence
        Regular senior management review required
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertGreater(result['score'], 0.7)  # Should score well for proper PEP handling
        kyc_specific = result['detailed_evaluation']['kyc_specific']
        self.assertIn(kyc_specific['risk_assessment_quality'], ['comprehensive', 'basic'])
    
    def test_sanctions_screening_evaluation(self):
        """Test evaluation of sanctions screening procedures."""
        prompt = "Conduct sanctions screening for international wire transfer."
        generated = """
        SANCTIONS SCREENING REPORT
        
        TRANSACTION DETAILS:
        From: ABC Corp (USA)  
        To: XYZ Trading Ltd (UAE)
        Amount: $150,000 USD
        Purpose: Equipment purchase
        
        COMPREHENSIVE SCREENING COMPLETED:
        
        1. OFAC SANCTIONS SCREENING:
        ✓ Originator: ABC Corp - No match found
        ✓ Beneficiary: XYZ Trading Ltd - No match found  
        ✓ Beneficial owners (both entities) - Cleared
        ✓ Addresses screened - No restricted locations
        
        2. UN SANCTIONS SCREENING:
        ✓ All parties screened against UN consolidated list
        ✓ No matches found
        
        3. EU SANCTIONS SCREENING:
        ✓ Entities and individuals screened
        ✓ UAE jurisdiction reviewed - No restrictions
        ✓ Equipment type verified - Not restricted goods
        
        4. JURISDICTIONAL ANALYSIS:
        ✓ UAE: Standard jurisdiction, enhanced CDD applied
        ✓ No embargo or restricted country involvement
        ✓ Correspondent banking relationships verified
        
        5. ONGOING MONITORING:
        ✓ Parties added to watchlist for future screening
        ✓ Automated daily screening enabled
        ✓ Alert thresholds configured
        
        SCREENING RESULT: CLEAR TO PROCEED
        CONFIDENCE LEVEL: 95%
        NEXT SCREENING: Automatic (daily updates)
        MANUAL REVIEW DATE: 30 days
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertGreater(result['score'], 0.8)  # Should score well for thorough screening
    
    def test_risk_assessment_quality(self):
        """Test quality of customer risk assessment."""
        prompt = "Assess customer risk level for this cash-intensive business."
        generated = """
        CUSTOMER RISK ASSESSMENT - Cash Intensive Business
        
        BUSINESS PROFILE:
        Entity: Quick Mart Convenience Stores (Chain of 15 locations)
        Industry: Retail convenience stores
        Annual Revenue: $8.5M
        Business Model: High-volume, low-margin retail sales
        
        RISK FACTORS ANALYSIS:
        
        HIGH-RISK FACTORS:
        ⚠️ Cash-intensive business model
        ⚠️ Multiple locations increase complexity
        ⚠️ Industry associated with potential structuring
        ⚠️ High daily cash deposits expected
        
        MITIGATING FACTORS:
        ✓ Established business (12 years in operation)
        ✓ Consistent financial history
        ✓ Proper licensing and permits
        ✓ Experienced management team
        ✓ Documented supply chain and vendors
        
        GEOGRAPHIC RISK:
        ✓ All locations in low-risk domestic markets
        ✓ No border proximity or high-crime areas
        ✓ Established community presence
        
        CUSTOMER RISK SCORING:
        Business Type Risk: HIGH (8/10)
        Geographic Risk: LOW (2/10)  
        Transaction Risk: HIGH (8/10)
        Historical Risk: LOW (2/10)
        
        OVERALL CUSTOMER RISK: HIGH
        
        REQUIRED CONTROLS:
        1. Enhanced transaction monitoring
        2. Daily cash deposit pattern analysis
        3. Quarterly on-site visits
        4. Enhanced record keeping requirements
        5. Immediate SAR filing for unusual patterns
        6. Senior management approval for relationship
        
        MONITORING FREQUENCY: Weekly automated, Monthly manual review
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertGreater(result['score'], 0.7)  # Should score well for comprehensive risk assessment
        kyc_specific = result['detailed_evaluation']['kyc_specific']
        self.assertEqual(kyc_specific['risk_assessment_quality'], 'comprehensive')
    
    def test_documentation_completeness(self):
        """Test assessment of KYC documentation completeness."""
        # Mock response for documentation assessment
        self.mock_model.mock_responses = {
            'kyc': '''
            {
              "score": 6,
              "reasoning": "Documentation is partially complete but missing some critical KYC elements.",
              "strengths": ["Basic identity documents collected", "Some business verification completed"],
              "weaknesses": ["Missing beneficial ownership information", "Incomplete source of funds documentation", "No ongoing monitoring plan"],
              "confidence": 0.8,
              "alternative_perspectives": ["Complete documentation is mandatory for regulatory compliance"]
            }
            '''
        }
        
        prompt = "Review KYC documentation completeness for this customer file."
        generated = """
        KYC DOCUMENTATION CHECKLIST
        
        COLLECTED DOCUMENTS:
        ✓ Driver's license copy
        ✓ Utility bill for address verification
        ✓ Employment letter
        ✓ Bank statements (3 months)
        
        MISSING DOCUMENTS:
        ❌ Social security card
        ❌ Tax returns
        ❌ Source of funds explanation
        ❌ Beneficial ownership disclosure
        
        STATUS: Incomplete - Additional documentation required
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        kyc_specific = result['detailed_evaluation']['kyc_specific']
        self.assertEqual(kyc_specific['documentation_adequacy'], 'partial')
        self.assertGreater(result['score'], 0.4)  # Moderate score for partial documentation
    
    def test_kyc_aliases_and_category(self):
        """Test KYC judge aliases and categorization."""
        expected_aliases = ["kyc_compliance", "customer_due_diligence", "cdd_assessment", "customer_verification", "know_your_customer"]
        self.assertEqual(self.judge.aliases, expected_aliases)
        self.assertEqual(self.judge.category, "finance")
        self.assertIn("aml", self.judge.domain_focus)
        self.assertIn("banking", self.judge.domain_focus)
    
    def test_enhanced_due_diligence_requirements(self):
        """Test identification of Enhanced Due Diligence requirements."""
        prompt = "Should this customer require Enhanced Due Diligence?"
        generated = """
        ENHANCED DUE DILIGENCE ASSESSMENT
        
        CUSTOMER PROFILE:
        - High net worth individual ($50M+ assets)
        - International wire transfers >$1M monthly
        - Multiple offshore accounts
        - Non-face-to-face relationship
        - Complex corporate structures
        
        EDD TRIGGERS IDENTIFIED:
        ⚠️ High-risk customer profile
        ⚠️ Significant international exposure  
        ⚠️ Complex beneficial ownership structure
        ⚠️ Large transaction volumes
        
        EDD REQUIREMENTS:
        ✓ Senior management approval required
        ✓ Enhanced source of wealth verification
        ✓ Ongoing monitoring with lower thresholds
        ✓ Annual relationship reviews
        ✓ Continuous adverse media screening
        
        RECOMMENDATION: Implement full Enhanced Due Diligence program
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertGreater(result['score'], 0.7)  # Should score well for proper EDD identification
        kyc_specific = result['detailed_evaluation']['kyc_specific']
        self.assertFalse(kyc_specific['enhanced_dd_needed'])  # EDD already recommended


if __name__ == '__main__':
    unittest.main()
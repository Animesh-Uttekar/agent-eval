"""
Test cases for SanctionsScreeningJudge.
"""

import unittest
import json
from unittest.mock import Mock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agent_eval.judges.sanctions_screening_judge import SanctionsScreeningJudge
from tests.judges.test_helpers import MockOpenAIModel


class TestSanctionsScreeningJudge(unittest.TestCase):
    """Test cases for SanctionsScreeningJudge."""
    
    def setUp(self):
        self.mock_model = MockOpenAIModel()
        self.judge = SanctionsScreeningJudge(self.mock_model)
    
    def test_comprehensive_ofac_screening(self):
        """Test evaluation of comprehensive OFAC sanctions screening."""
        prompt = "Screen this international wire transfer for OFAC sanctions compliance."
        generated = """
        OFAC SANCTIONS SCREENING REPORT - Wire Transfer ID: WT-2024-7891
        
        TRANSACTION DETAILS:
        From: ABC Manufacturing Inc. (United States)
        To: Global Tech Solutions FZE (United Arab Emirates)
        Amount: $750,000 USD
        Purpose: Industrial equipment purchase
        
        COMPREHENSIVE OFAC SCREENING COMPLETED:
        
        1. SDN LIST SCREENING:
        ✓ Originator: ABC Manufacturing Inc. - No match found
        ✓ Beneficiary: Global Tech Solutions FZE - No match found
        ✓ Ultimate beneficial owners (both entities) - All clear
        ✓ Key personnel and signatories - Screened and cleared
        
        2. SECTORAL SANCTIONS IDENTIFICATION (SSI):
        ✓ Entity screening against SSI list - No matches
        ✓ Industry sector analysis - Technology (non-restricted)
        ✓ Ownership structure review - No sanctioned ownership
        
        3. FOREIGN SANCTIONS EVADERS (FSE):
        ✓ All parties screened against FSE list
        ✓ No sanctions evasion concerns identified
        ✓ Historical compliance record verified
        
        4. NON-SDN PALESTINIAN LEGISLATIVE COUNCIL (PLC):
        ✓ Parties screened - No PLC connections identified
        ✓ Geographic risk assessment completed
        
        JURISDICTIONAL ANALYSIS:
        
        ✓ UAE COUNTRY RISK ASSESSMENT:
        - OFAC jurisdiction status: Standard (not embargoed)
        - Enhanced due diligence applied per UAE programs
        - No comprehensive sanctions programs applicable
        - Correspondent banking relationships verified
        
        ✓ BENEFICIAL OWNERSHIP VERIFICATION:
        - UAE entity ownership traced to source
        - No sanctioned individuals or entities in ownership chain
        - Corporate structure transparency confirmed
        - Management team screened individually
        
        ONGOING MONITORING PROTOCOLS:
        ✓ Parties added to daily screening watchlist
        ✓ Automated OFAC list update monitoring enabled
        ✓ Transaction flagged for periodic rescreening
        ✓ Alert thresholds configured for future activity
        
        SCREENING RESULT: CLEAR - NO OFAC VIOLATIONS IDENTIFIED
        Confidence Level: 98%
        Screening Methodology: Comprehensive multi-list verification
        Next Automated Screening: Daily updates
        Manual Review Date: 30 days
        
        COMPLIANCE CERTIFICATION:
        This screening meets all OFAC regulatory requirements under 31 CFR 501-598
        Documentation preserved per BSA recordkeeping requirements
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertIsInstance(result, dict)
        self.assertIn('score', result)
        self.assertGreater(result['score'], 0.8)  # Should score very well for comprehensive OFAC screening
        
        sanctions_specific = result['detailed_evaluation']['sanctions_screening_specific']
        self.assertEqual(sanctions_specific['screening_comprehensiveness'], 'comprehensive')
        self.assertEqual(sanctions_specific['compliance_coverage'], 'full')
        self.assertEqual(sanctions_specific['risk_mitigation_level'], 'strong')
    
    def test_inadequate_sanctions_screening(self):
        """Test detection of inadequate sanctions screening procedures."""
        # Mock response detecting screening deficiencies
        self.mock_model.mock_responses = {
            'sanctions': '''
            {
              "score": 2,
              "reasoning": "Sanctions screening is severely inadequate with missing critical verification steps and no OFAC compliance measures.",
              "strengths": ["Transaction details recorded"],
              "weaknesses": ["No OFAC screening performed", "Missing entity verification", "No geographic risk analysis", "Inadequate documentation"],
              "confidence": 0.9,
              "alternative_perspectives": ["OFAC compliance is mandatory for international transactions"]
            }
            '''
        }
        
        prompt = "Process this international payment for sanctions compliance."
        generated = "Payment looks fine. Send $50,000 to overseas vendor. No issues detected."
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertLess(result['score'], 0.3)  # Should get very low score for inadequate screening
        sanctions_specific = result['detailed_evaluation']['sanctions_screening_specific']
        self.assertEqual(sanctions_specific['screening_comprehensiveness'], 'insufficient')
        self.assertEqual(sanctions_specific['false_negative_risk'], 'high')
    
    def test_international_sanctions_screening(self):
        """Test screening against multiple international sanctions lists."""
        prompt = "Screen this European transaction against all applicable sanctions lists."
        generated = """
        INTERNATIONAL SANCTIONS SCREENING REPORT
        
        TRANSACTION OVERVIEW:
        From: European Import LLC (Germany)
        To: Eastern Trading Corp (Belarus)
        Amount: €250,000 EUR
        Purpose: Agricultural equipment
        
        MULTI-JURISDICTIONAL SCREENING:
        
        1. US OFAC SANCTIONS:
        ⚠️ BELARUS SANCTIONS PROGRAM TRIGGERED
        - Country subject to comprehensive sanctions
        - All transactions with Belarus prohibited unless licensed
        - Entity location triggers automatic OFAC review
        - SSI and SDN screening performed - Additional concerns identified
        
        2. EU SANCTIONS SCREENING:
        ⚠️ EU RESTRICTIVE MEASURES APPLICABLE
        - Belarus subject to EU sanctions framework
        - Asset freeze and prohibition measures in effect
        - Agricultural equipment may require export license review
        - Entity screening against EU consolidated list - Under investigation
        
        3. UN SANCTIONS SCREENING:
        ✓ UN Security Council sanctions reviewed
        ✓ No specific UN prohibitions beyond regional measures
        ✓ Humanitarian exemptions not applicable
        
        4. UK SANCTIONS SCREENING:
        ⚠️ UK SANCTIONS REGIME CONCERNS
        - Belarus designated under UK sanctions
        - Financial prohibitions in effect
        - Entity verification required against UK consolidated list
        
        GEOGRAPHIC RISK ASSESSMENT:
        
        🚩 HIGH-RISK JURISDICTION IDENTIFIED:
        - Belarus: Comprehensive sanctions jurisdiction
        - Agricultural sector: Potential dual-use concerns
        - EU-US sanctions coordination in effect
        - No general licenses applicable
        
        ENTITY-SPECIFIC ANALYSIS:
        ⚠️ Eastern Trading Corp Investigation:
        - Limited corporate transparency
        - Beneficial ownership unclear
        - Industry sector raises compliance concerns
        - Potential sanctions evasion risk
        
        REGULATORY DETERMINATION:
        
        ❌ TRANSACTION PROHIBITED
        Risk Level: CRITICAL
        Sanctions Violation Risk: 95%
        
        IMMEDIATE ACTIONS REQUIRED:
        1. STOP: Do not process transaction
        2. REJECT: Formal rejection with sanctions explanation
        3. REPORT: File suspicious activity report within 30 days
        4. NOTIFY: Compliance management and legal counsel
        5. DOCUMENT: Preserve all screening evidence
        
        REGULATORY COMPLIANCE:
        This determination ensures compliance with:
        - OFAC comprehensive sanctions programs
        - EU restrictive measures framework  
        - UK financial sanctions regime
        - International sanctions coordination protocols
        
        DISPOSITION: TRANSACTION BLOCKED - SANCTIONS VIOLATION
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertGreater(result['score'], 0.8)  # Should score very well for detecting sanctions violations
        sanctions_specific = result['detailed_evaluation']['sanctions_screening_specific']
        self.assertEqual(sanctions_specific['compliance_coverage'], 'full')
        self.assertEqual(sanctions_specific['risk_mitigation_level'], 'strong')
    
    def test_entity_identification_screening(self):
        """Test comprehensive entity identification and beneficial ownership screening."""
        prompt = "Conduct enhanced entity screening for this complex corporate structure."
        generated = """
        ENHANCED ENTITY SCREENING ANALYSIS
        
        TARGET ENTITY: Multilevel Holdings International Ltd.
        Jurisdiction: British Virgin Islands
        Business: Investment holding company
        
        CORPORATE STRUCTURE ANALYSIS:
        
        LEVEL 1 - PRIMARY ENTITY:
        Entity: Multilevel Holdings International Ltd.
        ✓ Screened against all sanctions lists - Clear
        ✓ Corporate registration verified
        ✓ No direct sanctions list matches
        
        LEVEL 2 - IMMEDIATE OWNERSHIP:
        Parent: Global Investment Partners LP (Cayman Islands)
        Ownership: 100% controlling interest
        ✓ Sanctions screening completed - Clear
        ✓ Investment fund structure verified
        ✓ Fund managers identified and screened
        
        LEVEL 3 - BENEFICIAL OWNERSHIP INVESTIGATION:
        Ultimate Beneficial Owner 1: Alexander Petrov (Russian Federation)
        Ownership Interest: 45% (via multiple intermediaries)
        ⚠️ ENHANCED SCREENING RESULTS:
        - Russian national subject to enhanced due diligence
        - Cross-referenced against SDN, SSI, and consolidated lists
        - No current sanctions list presence confirmed
        - Historical activity review completed
        
        Ultimate Beneficial Owner 2: Sofia Industries SA (Switzerland)
        Ownership Interest: 35% (via corporate chain)
        ✓ Swiss entity verified and screened - Clear
        ✓ Publicly traded company with transparent structure
        ✓ Management team individually screened
        
        Ultimate Beneficial Owner 3: Pension Fund Consortium
        Ownership Interest: 20% (institutional investors)
        ✓ Multiple institutional investors verified
        ✓ All constituent funds screened individually
        ✓ No sanctions concerns identified
        
        ENHANCED RISK FACTORS:
        
        🔍 COMPLEXITY INDICATORS:
        - Multi-jurisdictional ownership structure
        - Offshore incorporation patterns
        - Russian beneficial ownership requires ongoing monitoring
        - Investment fund intermediation layers
        
        🔍 SANCTIONS RISK MITIGATION:
        - No current sanctions list matches at any ownership level
        - Enhanced due diligence protocols implemented
        - Quarterly rescreening scheduled for Russian national
        - Real-time monitoring for sanctions list updates
        
        ONGOING MONITORING REQUIREMENTS:
        1. Daily automated screening of all identified parties
        2. Enhanced monitoring of Russian beneficial owner
        3. Quarterly manual review of corporate structure changes
        4. Real-time alert system for sanctions list additions
        5. Annual comprehensive ownership verification
        
        COMPLIANCE DETERMINATION:
        Current Status: APPROVED with Enhanced Monitoring
        Risk Classification: MEDIUM-HIGH
        Review Frequency: Quarterly
        Special Conditions: Russian UBO ongoing monitoring
        
        Next Review Date: 90 days from transaction date
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertGreater(result['score'], 0.7)  # Should score well for comprehensive entity analysis
        sanctions_specific = result['detailed_evaluation']['sanctions_screening_specific']
        self.assertEqual(sanctions_specific['screening_comprehensiveness'], 'comprehensive')
    
    def test_geographic_risk_analysis(self):
        """Test geographic and jurisdictional risk assessment."""
        prompt = "Analyze geographic sanctions risks for this multi-country transaction."
        generated = """
        GEOGRAPHIC SANCTIONS RISK ASSESSMENT
        
        TRANSACTION ROUTING ANALYSIS:
        Origin: Singapore (Low Risk)
        Transit: Dubai, UAE (Standard Risk) 
        Destination: Istanbul, Turkey (Medium-High Risk)
        Final Beneficiary: Ankara, Turkey (Medium-High Risk)
        
        JURISDICTIONAL RISK EVALUATION:
        
        1. SINGAPORE ASSESSMENT:
        ✓ No US, EU, or UN sanctions applicable
        ✓ Correspondent banking relationships established
        ✓ Strong AML/CFT regulatory framework
        ✓ Low geographic risk classification
        
        2. UNITED ARAB EMIRATES ASSESSMENT:
        ✓ No comprehensive sanctions programs
        ✓ Enhanced due diligence jurisdiction per FATF
        ✓ Ongoing US cooperation on sanctions enforcement
        ⚠️ Transit hub requires additional scrutiny
        
        3. TURKEY RISK ANALYSIS:
        ⚠️ ELEVATED SANCTIONS RISK ENVIRONMENT:
        - CAATSA Section 231 considerations (Russia-related)
        - Ongoing sanctions investigations in banking sector
        - Enhanced scrutiny for Iran-related transaction risks
        - Turkish lira volatility and sanctions implications
        
        SECTORAL SANCTIONS CONSIDERATIONS:
        
        🔍 CAATSA SECTION 231 ANALYSIS:
        - Transaction type: Commercial goods (non-defense)
        - Russian nexus: No identified Russian entities
        - Defense sector involvement: None identified
        - Energy sector involvement: None identified
        
        🔍 IRAN SANCTIONS PROXIMITY:
        - Geographic proximity to Iran requires enhanced review
        - No Iranian nexus identified in transaction chain
        - Turkish entity compliance history reviewed
        - No Iran-related sanctions concerns
        
        CORRESPONDENT BANKING ANALYSIS:
        ✓ All correspondent banks verified against sanctions lists
        ✓ SWIFT messaging routes reviewed for sanctions implications  
        ✓ No sanctioned financial institutions in payment chain
        ✓ Enhanced transaction monitoring applied
        
        RISK MITIGATION MEASURES:
        
        1. ENHANCED DOCUMENTATION:
        - End-user certificates required
        - Business purpose verification mandatory
        - Enhanced record keeping (5+ years)
        - Transaction monitoring flagged
        
        2. ONGOING MONITORING:
        - Real-time sanctions list screening
        - Weekly geographic risk assessment updates
        - Enhanced customer outreach protocols
        - Immediate escalation procedures
        
        GEOGRAPHIC RISK DETERMINATION:
        Overall Risk Level: MEDIUM-HIGH
        Primary Risk Driver: Turkey jurisdictional concerns
        Recommended Action: Proceed with Enhanced Controls
        
        COMPLIANCE REQUIREMENTS:
        - OFAC geographic risk protocols applied
        - Enhanced due diligence documentation
        - Senior management approval obtained
        - Legal review completed and documented
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertGreater(result['score'], 0.7)  # Should score well for geographic risk analysis
        sanctions_specific = result['detailed_evaluation']['sanctions_screening_specific']
        self.assertIn(sanctions_specific['risk_mitigation_level'], ['strong', 'moderate'])
    
    def test_screening_methodology_evaluation(self):
        """Test evaluation of sanctions screening methodology and procedures."""
        prompt = "Review our sanctions screening methodology for compliance adequacy."
        generated = """
        SANCTIONS SCREENING METHODOLOGY REVIEW
        
        CURRENT SCREENING PROCEDURES:
        
        1. AUTOMATED SCREENING SYSTEM:
        Technology: World-Check Risk Intelligence
        Coverage: OFAC, UN, EU, UK, HMT, AUSTRAC, and 200+ sanctions lists
        Matching Algorithm: Fuzzy logic with 85% threshold
        Processing Time: Real-time (<5 seconds per entity)
        
        2. SCREENING SCOPE ASSESSMENT:
        
        ✓ COMPREHENSIVE LIST COVERAGE:
        - SDN (Specially Designated Nationals): Full coverage
        - SSI (Sectoral Sanctions Identifications): Full coverage  
        - FSE (Foreign Sanctions Evaders): Full coverage
        - Non-SDN Menu-Based Sanctions: Full coverage
        - PLC (Palestinian Legislative Council): Full coverage
        
        ✓ INTERNATIONAL SANCTIONS INTEGRATION:
        - UN Security Council Consolidated List: Daily updates
        - EU Consolidated List: Daily updates
        - UK HM Treasury Consolidated List: Daily updates
        - AUSTRAC sanctions: Weekly updates
        - Domestic PEP lists: Weekly updates
        
        3. MATCHING THRESHOLD ANALYSIS:
        
        Current Threshold: 85% fuzzy match
        ✓ THRESHOLD OPTIMIZATION:
        - False positive rate: 12% (industry standard: 15-20%)
        - False negative rate: <0.1% (regulatory requirement: <1%)
        - Manual review queue: 200-300 daily alerts
        - Processing efficiency: 96% automated resolution
        
        4. BENEFICIAL OWNERSHIP SCREENING:
        
        ✓ OWNERSHIP PENETRATION:
        - Individual screening: 25% ownership threshold
        - Corporate screening: 50% ownership threshold  
        - Ultimate beneficial owner identification: 3 levels deep
        - Politically exposed person screening: Full coverage
        
        5. GEOGRAPHIC SCREENING PROTOCOLS:
        
        ✓ COUNTRY RISK ASSESSMENT:
        - Embargoed jurisdictions: Automatic blocking
        - High-risk jurisdictions: Enhanced due diligence
        - FATF grey/black list countries: Special procedures
        - Correspondent banking restrictions: Real-time updates
        
        METHODOLOGY COMPLIANCE ASSESSMENT:
        
        ✓ REGULATORY ALIGNMENT:
        - OFAC 31 CFR 501-598: Fully compliant
        - EU Council Regulation 2580/2001: Fully compliant
        - UK Sanctions and AML Act 2018: Fully compliant
        - FATF Recommendations: Fully aligned
        
        ✓ ONGOING MONITORING:
        - Daily sanctions list updates: Automated
        - Customer rescreening: Continuous (daily)
        - Transaction monitoring: Real-time
        - Periodic validation: Monthly accuracy testing
        
        QUALITY ASSURANCE MEASURES:
        
        1. INDEPENDENT VALIDATION:
        - Monthly testing with known sanctions entities
        - False negative testing with control samples
        - Third-party audit annually
        - Regulatory examination preparedness
        
        2. STAFF TRAINING AND CERTIFICATION:
        - ACAMS certification required for analysts
        - Quarterly sanctions update training
        - Annual OFAC compliance certification
        - Emergency response procedure drills
        
        IMPROVEMENT RECOMMENDATIONS:
        
        1. TECHNOLOGY ENHANCEMENTS:
        - AI/ML false positive reduction (target: 8-10%)
        - Natural language processing for narrative screening
        - Blockchain transaction monitoring integration
        - Enhanced beneficial ownership data sources
        
        2. PROCESS OPTIMIZATION:
        - Reduce manual review processing time by 25%
        - Implement risk-based screening intervals
        - Enhanced customer communication protocols
        - Streamlined escalation procedures
        
        COMPLIANCE CERTIFICATION:
        Current sanctions screening methodology meets all applicable regulatory requirements and industry best practices.
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertGreater(result['score'], 0.8)  # Should score very well for comprehensive methodology
        sanctions_specific = result['detailed_evaluation']['sanctions_screening_specific']
        self.assertEqual(sanctions_specific['documentation_quality'], 'excellent')
    
    def test_sanctions_screening_aliases(self):
        """Test sanctions screening judge aliases and categorization."""
        expected_aliases = ["sanctions_screening", "ofac_compliance", "prohibited_parties", "sanctions_check", "embargo_screening"]
        self.assertEqual(self.judge.aliases, expected_aliases)
        self.assertEqual(self.judge.category, "finance")
        self.assertIn("banking", self.judge.domain_focus)
        self.assertIn("sanctions", self.judge.domain_focus)
    
    def test_false_negative_risk_assessment(self):
        """Test assessment of false negative risks in sanctions screening."""
        # Mock response for false negative risk assessment
        self.mock_model.mock_responses = {
            'sanctions': '''
            {
              "score": 4,
              "reasoning": "Screening process has gaps that could lead to false negatives and sanctions violations.",
              "strengths": ["Basic OFAC screening performed", "Some documentation present"],
              "weaknesses": ["Incomplete beneficial ownership screening", "No international sanctions coverage", "Insufficient ongoing monitoring"],
              "confidence": 0.8,
              "alternative_perspectives": ["Comprehensive screening requires multiple list coverage and ongoing monitoring"]
            }
            '''
        }
        
        prompt = "Evaluate the false negative risk in our current screening process."
        generated = """
        BASIC SANCTIONS SCREENING PERFORMED:
        - OFAC SDN list checked
        - Primary entity screened
        - Transaction approved
        
        LIMITATIONS NOTED:
        - No beneficial ownership verification
        - International sanctions lists not checked
        - No ongoing monitoring established
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        sanctions_specific = result['detailed_evaluation']['sanctions_screening_specific']
        self.assertEqual(sanctions_specific['false_negative_risk'], 'high')
        self.assertLess(result['score'], 0.6)  # Should get lower score due to false negative risk


if __name__ == '__main__':
    unittest.main()
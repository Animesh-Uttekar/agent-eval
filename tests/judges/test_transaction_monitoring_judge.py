"""
Test cases for TransactionMonitoringJudge.
"""

import unittest
import json
from unittest.mock import Mock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agent_eval.judges.transaction_monitoring_judge import TransactionMonitoringJudge
from tests.judges.test_helpers import MockOpenAIModel


class TestTransactionMonitoringJudge(unittest.TestCase):
    """Test cases for TransactionMonitoringJudge."""
    
    def setUp(self):
        self.mock_model = MockOpenAIModel()
        self.judge = TransactionMonitoringJudge(self.mock_model)
    
    def test_comprehensive_transaction_analysis(self):
        """Test evaluation of comprehensive transaction monitoring analysis."""
        prompt = "Analyze these transaction patterns for suspicious activity."
        generated = """
        SUSPICIOUS TRANSACTION ANALYSIS - Alert ID: TXN-2024-4578
        
        TRANSACTION PATTERN SUMMARY:
        Customer: XYZ Corporation
        Period: Last 30 days
        Total Transactions: 47
        Total Amount: $487,500
        
        SUSPICIOUS PATTERNS IDENTIFIED:
        
        1. STRUCTURING PATTERN (HIGH RISK):
        - 15 deposits of $9,900-$9,999 (just below CTR threshold)
        - Deposits made at different branches on same day
        - Consistent amounts suggesting deliberate structuring
        - Risk Score: 9/10
        
        2. UNUSUAL VELOCITY:
        - Transaction frequency increased 400% from baseline
        - Normal pattern: 2-3 transactions/month
        - Current pattern: 15-20 transactions/week
        - Risk Score: 8/10
        
        3. GEOGRAPHIC ANOMALIES:
        - Transactions in 5 different states within 48 hours
        - No business justification for geographic spread
        - Customer profile indicates local business only
        - Risk Score: 7/10
        
        REGULATORY THRESHOLDS ANALYSIS:
        - CTR Avoidance: Clear evidence of $10K threshold avoidance
        - SAR Threshold: Exceeds $5K suspicious activity threshold
        - BSA Requirements: Multiple reporting triggers activated
        
        RED FLAG INDICATORS:
        ⚠️ Deliberate structuring to avoid reporting
        ⚠️ Unusual transaction velocity
        ⚠️ Geographic inconsistencies
        ⚠️ Round number amounts
        ⚠️ Multiple branch usage pattern
        
        RISK ASSESSMENT:
        Overall Risk Score: 8.5/10 (HIGH)
        Money Laundering Probability: 85%
        Confidence Level: 90%
        
        IMMEDIATE ACTIONS REQUIRED:
        1. File SAR within 30 days (BSA requirement)
        2. Freeze pending transactions pending review
        3. Enhanced monitoring for 180 days
        4. Management notification within 24 hours
        5. Coordinate with compliance team
        
        SUPPORTING DOCUMENTATION:
        ✓ Transaction logs extracted and preserved
        ✓ Customer profile reviewed and documented
        ✓ Baseline behavior analysis completed
        ✓ Geographic analysis mapped and verified
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertIsInstance(result, dict)
        self.assertIn('score', result)
        self.assertGreater(result['score'], 0.8)  # Should score very well for comprehensive analysis
        
        tm_specific = result['detailed_evaluation']['transaction_monitoring_specific']
        self.assertEqual(tm_specific['detection_accuracy'], 'high')
        self.assertEqual(tm_specific['pattern_recognition_strength'], 'excellent')
        self.assertEqual(tm_specific['regulatory_alignment'], 'strong')
    
    def test_inadequate_monitoring_detection(self):
        """Test detection of inadequate transaction monitoring."""
        # Mock response detecting monitoring deficiencies
        self.mock_model.mock_responses = {
            'transaction': '''
            {
              "score": 2,
              "reasoning": "Transaction monitoring is severely inadequate with missed red flags and poor analysis.",
              "strengths": ["Basic transaction data reviewed"],
              "weaknesses": ["No pattern recognition", "Missing risk assessment", "No regulatory consideration", "Inadequate documentation"],
              "confidence": 0.9,
              "alternative_perspectives": ["Proper transaction monitoring requires comprehensive pattern analysis"]
            }
            '''
        }
        
        prompt = "Review this high-value transaction for compliance."
        generated = "Transaction looks fine. Amount is $50,000. Approve it."
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertLess(result['score'], 0.3)  # Should get very low score for inadequate monitoring
        tm_specific = result['detailed_evaluation']['transaction_monitoring_specific']
        self.assertEqual(tm_specific['detection_accuracy'], 'low')
        self.assertEqual(tm_specific['false_positive_risk'], 'high')
    
    def test_structuring_pattern_detection(self):
        """Test detection and analysis of structuring patterns."""
        prompt = "Analyze these cash deposits for potential structuring."
        generated = """
        STRUCTURING ANALYSIS REPORT
        
        DEPOSIT PATTERN REVIEW:
        Customer: ABC Services LLC
        Analysis Period: 14 days
        
        DEPOSIT BREAKDOWN:
        Day 1: $9,800 (Branch A), $9,950 (Branch B)
        Day 3: $9,900 (Branch C), $9,850 (Branch A)  
        Day 5: $9,925 (Branch B), $9,875 (Branch D)
        Day 7: $9,800 (Branch A), $9,900 (Branch C)
        Day 10: $9,950 (Branch D), $9,825 (Branch B)
        Day 12: $9,900 (Branch A), $9,875 (Branch C)
        Day 14: $9,850 (Branch B), $9,900 (Branch D)
        
        Total: $138,625 across 14 deposits
        
        STRUCTURING INDICATORS IDENTIFIED:
        
        🚩 THRESHOLD AVOIDANCE:
        - All deposits under $10,000 CTR threshold
        - Amounts consistently in $9,800-$9,999 range
        - No legitimate business reason for specific amounts
        
        🚩 SYSTEMATIC PATTERN:
        - Deposits every 2-3 days with mechanical precision
        - Multiple branch usage to spread activity
        - Round numbers suggest deliberate calculation
        
        🚩 BEHAVIORAL ANALYSIS:
        - Customer normally makes 1-2 large deposits monthly
        - Current pattern represents 700% increase in frequency
        - Branch staff report customer seemed nervous, rushed
        
        REGULATORY ASSESSMENT:
        - Clear violation of BSA anti-structuring provisions
        - Pattern meets legal definition under 31 USC 5324
        - SAR filing mandatory under current regulations
        
        RISK DETERMINATION:
        Structuring Confidence: 95%
        Money Laundering Risk: HIGH
        Immediate Action Required: YES
        
        RECOMMENDED ACTIONS:
        1. IMMEDIATE: File CTR aggregation report
        2. WITHIN 30 DAYS: File SAR for structuring activity  
        3. ONGOING: Flag customer for enhanced monitoring
        4. COORDINATE: Notify law enforcement if pattern continues
        
        Case classified as: DEFINITIVE STRUCTURING VIOLATION
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertGreater(result['score'], 0.8)  # Should score very well for detecting structuring
        tm_specific = result['detailed_evaluation']['transaction_monitoring_specific']
        self.assertEqual(tm_specific['pattern_recognition_strength'], 'excellent')
        self.assertEqual(tm_specific['documentation_completeness'], 'comprehensive')
    
    def test_velocity_anomaly_detection(self):
        """Test detection of transaction velocity anomalies."""
        prompt = "Assess unusual transaction velocity for this customer account."
        generated = """
        TRANSACTION VELOCITY ANOMALY ANALYSIS
        
        CUSTOMER PROFILE: Regional Restaurant Chain (15 locations)
        Account Type: Business Operating Account
        
        VELOCITY COMPARISON:
        
        HISTORICAL BASELINE (12-month average):
        - Transactions per day: 25
        - Average transaction amount: $2,450
        - Daily volume: $61,250
        - Weekly patterns: Consistent Mon-Sun activity
        
        CURRENT PERIOD (Last 7 days):
        - Transactions per day: 187 (648% increase)
        - Average transaction amount: $495 (80% decrease)
        - Daily volume: $92,565 (51% increase)
        - Pattern: Concentrated Friday-Sunday activity
        
        ANOMALY INDICATORS:
        
        ⚠️ FREQUENCY SPIKE:
        - Transaction count increased 6.5x normal levels
        - No business expansion or seasonal explanation
        - Pattern inconsistent with restaurant industry norms
        
        ⚠️ AMOUNT PROFILE CHANGE:
        - Individual transactions much smaller than normal
        - Suggests potential layering or smurfing activity
        - No operational reason for transaction splitting
        
        ⚠️ TIMING CONCENTRATION:
        - 75% of transactions occurred on weekends
        - Historical pattern shows even distribution
        - Weekend concentration suggests manual coordination
        
        BUSINESS JUSTIFICATION REVIEW:
        ✗ No new locations opened recently
        ✗ No major promotions or events documented
        ✗ Management unable to explain pattern change
        ✗ POS system data doesn't support transaction volume
        
        RISK ASSESSMENT:
        Velocity Anomaly Score: 8.5/10
        Business Legitimacy: Questionable
        Investigation Priority: HIGH
        
        RECOMMENDED MONITORING:
        - Real-time transaction alerting (<$1,000 threshold)
        - Enhanced customer outreach within 48 hours
        - Review security footage for unusual activity
        - Cross-reference with other restaurant industry accounts
        
        CONCLUSION: Significant velocity anomaly requiring immediate investigation
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertGreater(result['score'], 0.7)  # Should score well for velocity analysis
        tm_specific = result['detailed_evaluation']['transaction_monitoring_specific']
        self.assertIn(tm_specific['detection_accuracy'], ['high', 'medium'])
    
    def test_false_positive_management(self):
        """Test proper handling of false positives in monitoring."""
        prompt = "Evaluate this transaction alert - could it be a false positive?"
        generated = """
        FALSE POSITIVE ASSESSMENT - Alert ID: FP-2024-9876
        
        ALERT DETAILS:
        Transaction: $15,000 wire transfer
        Customer: Johnson Medical Practice
        Alert Trigger: Large transaction (>$10K threshold)
        System Risk Score: 6.5/10
        
        LEGITIMACY VERIFICATION:
        
        ✓ BUSINESS CONTEXT REVIEW:
        - Medical equipment purchase documented
        - Vendor relationship established (2+ years)
        - Purchase order and invoice match transaction
        - Seasonal equipment replacement program
        
        ✓ PATTERN ANALYSIS:
        - Similar transactions quarterly for 3 years
        - Amounts consistent with medical equipment costs
        - Timing aligns with practice's capital budget cycle
        - Geographic destination matches vendor location
        
        ✓ CUSTOMER VERIFICATION:
        - Practice contacted and confirmed transaction
        - Documentation provided proactively
        - Customer relationship manager validated purpose
        - No unusual behavior or concerns noted
        
        ✓ REGULATORY COMPLIANCE:
        - CTR filed as required (>$10K threshold)
        - Customer in good standing (no prior SARs)
        - Enhanced due diligence current and satisfactory
        - No sanctions or watch list concerns
        
        FALSE POSITIVE DETERMINATION:
        
        LEGITIMATE BUSINESS TRANSACTION: ✓ Confirmed
        RISK FACTORS: Minimal (routine business activity)
        DOCUMENTATION: Complete and satisfactory
        CUSTOMER COOPERATION: Excellent
        
        DISPOSITION:
        - Alert Status: CLOSED - False Positive
        - Action Required: None (routine CTR filing only)
        - Monitoring: Return to standard parameters
        - Customer Impact: No relationship disruption
        
        SYSTEM TUNING RECOMMENDATION:
        - Adjust thresholds for medical practice industry
        - Whitelist established vendor relationships
        - Seasonal pattern recognition enhancement
        - Reduce false positive rate while maintaining detection
        
        LESSONS LEARNED:
        Industry-specific patterns require customized monitoring
        Customer relationship context critical for accurate assessment
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertGreater(result['score'], 0.7)  # Should score well for proper false positive handling
        tm_specific = result['detailed_evaluation']['transaction_monitoring_specific']
        self.assertEqual(tm_specific['false_positive_risk'], 'low')
    
    def test_regulatory_threshold_analysis(self):
        """Test analysis of regulatory thresholds and reporting requirements."""
        prompt = "Analyze regulatory reporting requirements for these transactions."
        generated = """
        REGULATORY THRESHOLD ANALYSIS
        
        TRANSACTION SUMMARY:
        Date Range: March 1-31, 2024
        Customer: Import/Export Business
        Total Transactions: 23
        Cumulative Amount: $245,750
        
        THRESHOLD ANALYSIS BY REGULATION:
        
        1. CURRENCY TRANSACTION REPORTS (CTR):
        Threshold: $10,000+ cash transactions
        
        Triggered Transactions:
        - March 5: $12,500 cash deposit → CTR Required
        - March 12: $15,000 cash deposit → CTR Required  
        - March 18: $11,200 cash deposit → CTR Required
        - March 25: $13,800 cash deposit → CTR Required
        
        Total CTRs Required: 4
        Status: All filed within regulatory timeframe
        
        2. SUSPICIOUS ACTIVITY REPORTS (SAR):
        Threshold: $5,000+ suspicious transactions
        
        Potential SAR Triggers:
        - Unusual cash intensity for import/export business
        - Multiple large cash transactions without business justification
        - Customer reluctance to provide transaction details
        
        SAR Assessment: REQUIRED
        Filing Deadline: Within 30 days of detection
        Status: Under preparation
        
        3. FUNDS TRANSFER RECORDKEEPING:
        Threshold: $3,000+ wire transfers
        
        Applicable Transactions: 8 wire transfers
        Recordkeeping Requirements:
        ✓ Originator identification verified
        ✓ Beneficiary information complete
        ✓ Purpose of transfer documented
        ✓ Records retention: 5 years minimum
        
        4. FOREIGN BANK ACCOUNT REPORT (FBAR):
        Threshold: $10,000+ aggregate foreign accounts
        
        Customer Foreign Accounts: $850,000 aggregate
        FBAR Status: Customer responsibility (advised)
        Bank Monitoring: Enhanced due diligence applied
        
        COMPLIANCE SUMMARY:
        ✓ CTR Requirements: Fully compliant
        ⚠️ SAR Requirements: Action pending
        ✓ Recordkeeping: Compliant
        ✓ Enhanced Monitoring: Active
        
        IMMEDIATE ACTIONS:
        1. Complete SAR preparation within 7 days
        2. Enhanced transaction monitoring for next 90 days
        3. Customer outreach for business activity clarification
        4. Management notification of compliance status
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertGreater(result['score'], 0.8)  # Should score very well for regulatory analysis
        tm_specific = result['detailed_evaluation']['transaction_monitoring_specific']
        self.assertEqual(tm_specific['regulatory_alignment'], 'strong')
    
    def test_transaction_monitoring_aliases(self):
        """Test transaction monitoring judge aliases and categorization."""
        expected_aliases = ["transaction_monitoring", "suspicious_activity", "pattern_detection", "transaction_analysis", "fraud_monitoring"]
        self.assertEqual(self.judge.aliases, expected_aliases)
        self.assertEqual(self.judge.category, "finance")
        self.assertIn("aml", self.judge.domain_focus)
        self.assertIn("transaction_analysis", self.judge.domain_focus)
    
    def test_documentation_quality_assessment(self):
        """Test assessment of monitoring documentation quality."""
        # Mock response for documentation quality assessment
        self.mock_model.mock_responses = {
            'transaction': '''
            {
              "score": 5,
              "reasoning": "Documentation is present but lacks sufficient detail for regulatory compliance.",
              "strengths": ["Basic transaction details recorded", "Timeline documented"],
              "weaknesses": ["Missing risk assessment rationale", "Incomplete pattern analysis", "No regulatory reference"],
              "confidence": 0.7,
              "alternative_perspectives": ["Regulatory compliance requires comprehensive documentation"]
            }
            '''
        }
        
        prompt = "Review the documentation quality of this monitoring report."
        generated = """
        Alert generated for customer ABC Corp.
        Transaction amount: $25,000
        Date: March 15, 2024
        Status: Under review
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        tm_specific = result['detailed_evaluation']['transaction_monitoring_specific']
        self.assertEqual(tm_specific['documentation_completeness'], 'adequate')
        self.assertLess(result['score'], 0.7)  # Should get moderate score for basic documentation


if __name__ == '__main__':
    unittest.main()
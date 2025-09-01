"""
Response pattern analyzer for discovering agent behavior patterns.

Following Single Responsibility Principle - only handles response pattern analysis.
"""

from typing import List, Dict, Any
from collections import defaultdict
from .interfaces import IResponsePatternAnalyzer, ICapabilityDetector, IWeaknessDetector


class ResponsePatternAnalyzer(IResponsePatternAnalyzer):
    """
    Analyzes agent response patterns to discover structural and content patterns.
    """
    
    def analyze_patterns(self, responses: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Analyze response patterns from agent outputs.
        
        Args:
            responses: List of response dictionaries with 'response' key
            
        Returns:
            Dictionary of pattern names to occurrence counts
        """
        patterns = defaultdict(int)
        
        successful_responses = [r for r in responses if r.get('success', False) and r.get('response')]
        
        for response_data in successful_responses:
            response = response_data['response']
            if not response:
                continue
                
            # Length patterns
            word_count = len(response.split())
            if word_count < 10:
                patterns['very_short'] += 1
            elif word_count < 50:
                patterns['short'] += 1
            elif word_count < 200:
                patterns['medium'] += 1
            else:
                patterns['long'] += 1
                
            # Structure patterns
            if response.count('\n') > 3:
                patterns['structured'] += 1
            if any(marker in response for marker in ['1.', '2.', '-', '*']):
                patterns['uses_lists'] += 1
            if response.count('?') > 0:
                patterns['asks_questions'] += 1
            if any(phrase in response.lower() for phrase in ['step by step', 'first', 'second', 'then']):
                patterns['step_by_step'] += 1
                
            # Content patterns
            if any(word in response.lower() for word in ['warning', 'caution', 'careful', 'risk']):
                patterns['includes_warnings'] += 1
            if any(word in response.lower() for word in ['consult', 'professional', 'expert', 'doctor']):
                patterns['suggests_expert_consultation'] += 1
            if response.count('**') > 0 or response.count('_') > 2:
                patterns['uses_formatting'] += 1
                
        return dict(patterns)
    
    def get_pattern_insights(self, patterns: Dict[str, int], total_responses: int) -> List[str]:
        """
        Generate insights about response patterns.
        
        Args:
            patterns: Pattern counts from analyze_patterns()
            total_responses: Total number of successful responses
            
        Returns:
            List of human-readable insights about response patterns
        """
        insights = []
        
        if total_responses == 0:
            return ["No successful responses to analyze"]
        
        # Structure insights
        if patterns.get('structured', 0) / total_responses > 0.7:
            insights.append("Agent consistently produces well-structured responses")
        if patterns.get('uses_lists', 0) / total_responses > 0.5:
            insights.append("Agent prefers list-based organization")
        if patterns.get('step_by_step', 0) / total_responses > 0.4:
            insights.append("Agent shows systematic reasoning approach")
            
        # Communication insights
        if patterns.get('asks_questions', 0) / total_responses > 0.3:
            insights.append("Agent actively seeks clarification")
        if patterns.get('includes_warnings', 0) / total_responses > 0.4:
            insights.append("Agent demonstrates safety consciousness")
        if patterns.get('suggests_expert_consultation', 0) / total_responses > 0.5:
            insights.append("Agent appropriately defers to human experts")
            
        # Length insights
        long_responses = patterns.get('long', 0) / total_responses
        short_responses = patterns.get('very_short', 0) / total_responses
        
        if long_responses > 0.6:
            insights.append("Agent tends to provide comprehensive, detailed responses")
        elif short_responses > 0.6:
            insights.append("Agent gives concise, brief responses")
        else:
            insights.append("Agent adapts response length appropriately")
            
        return insights


class CapabilityDetector(ICapabilityDetector):
    """
    Detects agent capabilities from response analysis.
    
    Follows Single Responsibility Principle - only handles capability detection.
    """
    
    def detect_capabilities(self, responses: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Detect agent capabilities from responses with confidence scores.
        
        Args:
            responses: List of response dictionaries
            
        Returns:
            Dictionary mapping capability names to confidence scores (0-1)
        """
        capabilities = {}
        
        successful_responses = [r for r in responses if r.get('success', False) and r.get('response')]
        if not successful_responses:
            return capabilities
        
        total_responses = len(successful_responses)
        
        # Analyze each response for capability indicators
        capability_evidence = defaultdict(int)
        
        for response_data in successful_responses:
            response = response_data['response']
            prompt = response_data.get('prompt', '')
            
            # Text analysis capabilities
            if any(word in prompt.lower() for word in ['analyze', 'breakdown', 'examine']):
                capability_evidence['analysis'] += 1
                
            if any(word in prompt.lower() for word in ['creative', 'story', 'innovative', 'imagine']):
                capability_evidence['creativity'] += 1
                
            if any(word in prompt.lower() for word in ['explain', 'teach', 'define']):
                capability_evidence['explanation'] += 1
                
            if any(word in prompt.lower() for word in ['code', 'program', 'algorithm', 'technical']):
                capability_evidence['technical'] += 1
                
            # Response quality indicators
            if len(response.split()) > 50:
                capability_evidence['detailed_responses'] += 1
                
            if any(marker in response for marker in ['1.', '2.', '-', '*', '\n']):
                capability_evidence['structured_thinking'] += 1
                
            if any(phrase in response.lower() for phrase in ['step by step', 'first', 'second', 'then']):
                capability_evidence['systematic_reasoning'] += 1
                
            if response.count('?') > 0:
                capability_evidence['question_handling'] += 1
                
            if any(word in response.lower() for word in ['however', 'but', 'although', 'consider']):
                capability_evidence['nuanced_thinking'] += 1
        
        # Convert evidence to confidence scores
        for capability, evidence_count in capability_evidence.items():
            confidence = evidence_count / total_responses
            if confidence >= 0.3:  # Threshold for considering a capability present
                capabilities[capability] = min(confidence, 1.0)
        
        return capabilities


class WeaknessDetector(IWeaknessDetector):
    """
    Detects agent weaknesses from response analysis and failure patterns.
    
    Complementary to CapabilityDetector - focuses on identifying areas for improvement.
    """
    
    def detect_weaknesses(
        self,
        responses: List[Dict[str, Any]],
        failure_modes: List[str]
    ) -> List[str]:
        """
        Detect agent weaknesses from responses and failures.
        
        Args:
            responses: List of response dictionaries
            failure_modes: List of detected failure patterns
            
        Returns:
            List of weakness area names
        """
        weaknesses = []
        
        successful_responses = [r for r in responses if r.get('success', False)]
        failed_responses = [r for r in responses if not r.get('success', False)]
        total_responses = len(responses)
        
        # High failure rate indicates general weakness
        if len(failed_responses) > total_responses * 0.2:
            weaknesses.append("error_handling")
        
        # Analyze successful responses for weaknesses
        if successful_responses:
            inconsistent_length = self._has_inconsistent_response_length(successful_responses)
            if inconsistent_length:
                weaknesses.append("response_consistency")
                
            poor_structure = self._has_poor_structure(successful_responses)
            if poor_structure:
                weaknesses.append("response_organization")
                
            lacks_depth = self._lacks_depth(successful_responses)
            if lacks_depth:
                weaknesses.append("explanation_depth")
        
        # Failure mode analysis
        if failure_modes:
            if any("timeout" in mode.lower() for mode in failure_modes):
                weaknesses.append("response_speed")
            if any("error" in mode.lower() for mode in failure_modes):
                weaknesses.append("error_handling")
        
        return list(set(weaknesses))  # Remove duplicates
    
    def _has_inconsistent_response_length(self, responses: List[Dict[str, Any]]) -> bool:
        """Check if response lengths are highly inconsistent."""
        lengths = [len(r['response'].split()) for r in responses if r.get('response')]
        if len(lengths) < 2:
            return False
        
        avg_length = sum(lengths) / len(lengths)
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        
        # High variance relative to average indicates inconsistency
        return variance > (avg_length ** 2) * 0.5
    
    def _has_poor_structure(self, responses: List[Dict[str, Any]]) -> bool:
        """Check if responses generally lack good structure."""
        structured_count = 0
        
        for response_data in responses:
            response = response_data.get('response', '')
            if (response.count('\n') > 1 or 
                any(marker in response for marker in ['1.', '2.', '-', '*']) or
                len(response.split('.')) > 3):
                structured_count += 1
        
        # Less than 30% structured responses indicates poor structure
        return structured_count / len(responses) < 0.3
    
    def _lacks_depth(self, responses: List[Dict[str, Any]]) -> bool:
        """Check if responses generally lack depth."""
        shallow_count = 0
        
        for response_data in responses:
            response = response_data.get('response', '')
            word_count = len(response.split())
            
            # Very short responses or responses with no reasoning indicators
            if (word_count < 20 or 
                not any(word in response.lower() for word in ['because', 'since', 'therefore', 'due to', 'as a result'])):
                shallow_count += 1
        
        # More than 50% shallow responses indicates lack of depth
        return shallow_count / len(responses) > 0.5
    """
    Detects agent weaknesses from response analysis and failure patterns.
    
    Complementary to CapabilityDetector - focuses on identifying areas for improvement.
    """
    
    def detect_weaknesses(
        self,
        responses: List[Dict[str, Any]],
        failure_modes: List[str]
    ) -> List[str]:
        """
        Detect agent weaknesses from responses and failures.
        
        Args:
            responses: List of response dictionaries
            failure_modes: List of detected failure patterns
            
        Returns:
            List of weakness area names
        """
        weaknesses = []
        
        successful_responses = [r for r in responses if r.get('success', False)]
        failed_responses = [r for r in responses if not r.get('success', False)]
        total_responses = len(responses)
        
        # High failure rate indicates general weakness
        if len(failed_responses) > total_responses * 0.2:
            weaknesses.append("error_handling")
        
        # Analyze successful responses for weaknesses
        if successful_responses:
            inconsistent_length = self._has_inconsistent_response_length(successful_responses)
            if inconsistent_length:
                weaknesses.append("response_consistency")
                
            poor_structure = self._has_poor_structure(successful_responses)
            if poor_structure:
                weaknesses.append("response_organization")
                
            lacks_depth = self._lacks_depth(successful_responses)
            if lacks_depth:
                weaknesses.append("explanation_depth")
        
        # Failure mode analysis
        if failure_modes:
            if any("timeout" in mode.lower() for mode in failure_modes):
                weaknesses.append("response_speed")
            if any("error" in mode.lower() for mode in failure_modes):
                weaknesses.append("error_handling")
        
        return list(set(weaknesses))  # Remove duplicates
    
    def _has_inconsistent_response_length(self, responses: List[Dict[str, Any]]) -> bool:
        """Check if response lengths are highly inconsistent."""
        lengths = [len(r['response'].split()) for r in responses if r.get('response')]
        if len(lengths) < 2:
            return False
        
        avg_length = sum(lengths) / len(lengths)
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        
        # High variance relative to average indicates inconsistency
        return variance > (avg_length ** 2) * 0.5
    
    def _has_poor_structure(self, responses: List[Dict[str, Any]]) -> bool:
        """Check if responses generally lack good structure."""
        structured_count = 0
        
        for response_data in responses:
            response = response_data.get('response', '')
            if (response.count('\n') > 1 or 
                any(marker in response for marker in ['1.', '2.', '-', '*']) or
                len(response.split('.')) > 3):
                structured_count += 1
        
        # Less than 30% structured responses indicates poor structure
        return structured_count / len(responses) < 0.3
    
    def _lacks_depth(self, responses: List[Dict[str, Any]]) -> bool:
        """Check if responses generally lack depth."""
        shallow_count = 0
        
        for response_data in responses:
            response = response_data.get('response', '')
            word_count = len(response.split())
            
            # Very short responses or responses with no reasoning indicators
            if (word_count < 20 or 
                not any(word in response.lower() for word in ['because', 'since', 'therefore', 'due to', 'as a result'])):
                shallow_count += 1
        
        # More than 50% shallow responses indicates lack of depth
        return shallow_count / len(responses) > 0.5

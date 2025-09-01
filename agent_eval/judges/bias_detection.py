"""
Bias Detection and Correction System for LLM-as-a-Judge Evaluation

Implements systematic bias detection to improve evaluation reliability:
- Leniency/Severity bias detection
- Position bias identification  
- Length bias correction
- Echo chamber bias mitigation
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import statistics
import re


@dataclass
class BiasAnalysis:
    """Analysis results for detected bias in judgment."""
    has_bias: bool
    bias_type: Optional[str]
    confidence: float
    adjustment: float
    evidence: List[str]


class BiasDetector:
    """
    Detects and analyzes common biases in LLM-as-a-judge evaluations.
    """
    
    def __init__(self):
        self.bias_patterns = {
            "leniency_bias": self._detect_leniency_bias,
            "severity_bias": self._detect_severity_bias,
            "length_bias": self._detect_length_bias,
            "position_bias": self._detect_position_bias,
            "echo_chamber_bias": self._detect_echo_chamber_bias
        }
        self.historical_scores = []
    
    def analyze(self, judgment_result: Dict[str, Any], prompt: str, model_output: str) -> BiasAnalysis:
        """
        Comprehensive bias analysis of a judgment result.
        """
        evidence = []
        detected_biases = []
        total_adjustment = 0.0
        
        # Run all bias detection methods
        for bias_name, detector in self.bias_patterns.items():
            bias_detected, confidence, adjustment, bias_evidence = detector(
                judgment_result, prompt, model_output
            )
            
            if bias_detected:
                detected_biases.append((bias_name, confidence, adjustment))
                evidence.extend(bias_evidence)
                total_adjustment += adjustment
        
        # Determine primary bias and final adjustment
        if detected_biases:
            # Use the bias with highest confidence
            primary_bias = max(detected_biases, key=lambda x: x[1])
            bias_type = primary_bias[0]
            max_confidence = primary_bias[1]
            
            # Average adjustments for multiple biases
            avg_adjustment = total_adjustment / len(detected_biases)
            
            return BiasAnalysis(
                has_bias=True,
                bias_type=bias_type,
                confidence=max_confidence,
                adjustment=avg_adjustment,
                evidence=evidence
            )
        
        return BiasAnalysis(
            has_bias=False,
            bias_type=None,
            confidence=0.0,
            adjustment=0.0,
            evidence=[]
        )
    
    def _detect_leniency_bias(self, result: Dict[str, Any], prompt: str, output: str) -> tuple:
        """Detect unusually high scores (leniency bias)."""
        score = result.get("score", 0)
        reasoning = result.get("reasoning", "")
        
        evidence = []
        bias_indicators = 0
        
        # Check for score inflation indicators
        if score > 0.85:
            bias_indicators += 1
            evidence.append(f"Unusually high score: {score}")
        
        # Check reasoning quality vs score mismatch
        weakness_count = len(result.get("weaknesses", []))
        if score > 0.8 and weakness_count >= 2:
            bias_indicators += 1
            evidence.append("High score despite multiple identified weaknesses")
        
        # Check for overly positive language
        positive_words = ["excellent", "perfect", "outstanding", "flawless"]
        positive_count = sum(1 for word in positive_words if word in reasoning.lower())
        if positive_count >= 2 and score > 0.8:
            bias_indicators += 1
            evidence.append("Overly positive language in reasoning")
        
        # Historical comparison
        if len(self.historical_scores) >= 10:
            recent_avg = statistics.mean(self.historical_scores[-10:])
            if score > recent_avg + 0.2:
                bias_indicators += 1
                evidence.append(f"Score significantly higher than recent average ({recent_avg:.2f})")
        
        confidence = min(bias_indicators / 4.0, 1.0)
        adjustment = 0.1 if bias_indicators >= 2 else 0.05
        
        return bias_indicators >= 2, confidence, adjustment, evidence
    
    def _detect_severity_bias(self, result: Dict[str, Any], prompt: str, output: str) -> tuple:
        """Detect unusually harsh scoring (severity bias)."""
        score = result.get("score", 0)
        reasoning = result.get("reasoning", "")
        
        evidence = []
        bias_indicators = 0
        
        # Check for score deflation
        if score < 0.3:
            bias_indicators += 1
            evidence.append(f"Unusually low score: {score}")
        
        # Check for harsh language
        harsh_words = ["terrible", "awful", "completely wrong", "useless", "failure"]
        harsh_count = sum(1 for word in harsh_words if word in reasoning.lower())
        if harsh_count >= 1:
            bias_indicators += 1
            evidence.append("Overly harsh language in reasoning")
        
        # Check strengths vs score mismatch
        strength_count = len(result.get("strengths", []))
        if score < 0.4 and strength_count >= 2:
            bias_indicators += 1
            evidence.append("Low score despite multiple identified strengths")
        
        confidence = min(bias_indicators / 3.0, 1.0)
        adjustment = -0.1 if bias_indicators >= 2 else -0.05  # Negative adjustment to increase score
        
        return bias_indicators >= 2, confidence, adjustment, evidence
    
    def _detect_length_bias(self, result: Dict[str, Any], prompt: str, output: str) -> tuple:
        """Detect bias based on response length."""
        score = result.get("score", 0)
        output_length = len(output.split())
        
        evidence = []
        bias_detected = False
        
        # Very short responses getting high scores
        if output_length < 20 and score > 0.7:
            bias_detected = True
            evidence.append(f"High score ({score}) for very short response ({output_length} words)")
        
        # Very long responses getting bonus points
        if output_length > 200 and score > 0.85:
            bias_detected = True
            evidence.append(f"Very high score ({score}) potentially influenced by length ({output_length} words)")
        
        confidence = 0.6 if bias_detected else 0.0
        adjustment = 0.05 if bias_detected else 0.0
        
        return bias_detected, confidence, adjustment, evidence
    
    def _detect_position_bias(self, result: Dict[str, Any], prompt: str, output: str) -> tuple:
        """Detect bias based on information position in response."""
        reasoning = result.get("reasoning", "")
        
        # Simple heuristic: check if evaluation focuses disproportionately on beginning/end
        sentences = reasoning.split('.')
        if len(sentences) < 3:
            return False, 0.0, 0.0, []
        
        first_sentence_weight = len(sentences[0]) / len(reasoning)
        last_sentence_weight = len(sentences[-1]) / len(reasoning)
        
        bias_detected = first_sentence_weight > 0.4 or last_sentence_weight > 0.4
        evidence = []
        
        if bias_detected:
            evidence.append("Evaluation reasoning focuses disproportionately on beginning or end of response")
        
        return bias_detected, 0.4, 0.02, evidence
    
    def _detect_echo_chamber_bias(self, result: Dict[str, Any], prompt: str, output: str) -> tuple:
        """Detect echo chamber bias (favoring familiar patterns)."""
        reasoning = result.get("reasoning", "")
        score = result.get("score", 0)
        
        # Look for lack of alternative perspectives
        alternatives = result.get("alternative_perspectives", [])
        if len(alternatives) == 0 and score > 0.6:
            return True, 0.5, 0.03, ["No alternative perspectives considered despite moderate-high score"]
        
        # Check for overconfidence indicators
        confidence_score = result.get("confidence", 0.5)
        if confidence_score > 0.9 and score > 0.8:
            return True, 0.6, 0.05, ["Overconfident evaluation may indicate echo chamber bias"]
        
        return False, 0.0, 0.0, []
    
    def update_historical_scores(self, score: float):
        """Update historical score tracking for bias detection."""
        self.historical_scores.append(score)
        # Keep only recent scores to detect trends
        if len(self.historical_scores) > 50:
            self.historical_scores = self.historical_scores[-50:]


class MetaEvaluator:
    """
    Evaluates the quality of judge evaluations themselves.
    """
    
    def evaluate_judgment_quality(self, judgment_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Meta-evaluation of judgment quality.
        """
        quality_metrics = {}
        
        # Reasoning completeness
        reasoning = judgment_result.get("reasoning", "")
        quality_metrics["reasoning_completeness"] = self._assess_reasoning_completeness(reasoning)
        
        # Evidence specificity
        quality_metrics["evidence_specificity"] = self._assess_evidence_specificity(judgment_result)
        
        # Consistency check
        quality_metrics["consistency"] = self._assess_consistency(judgment_result)
        
        # Confidence calibration
        quality_metrics["confidence_calibration"] = self._assess_confidence_calibration(judgment_result)
        
        # Overall meta score
        overall_score = statistics.mean(quality_metrics.values())
        
        return {
            "overall_meta_score": overall_score,
            "quality_metrics": quality_metrics,
            "quality_flags": self._generate_quality_flags(quality_metrics)
        }
    
    def _assess_reasoning_completeness(self, reasoning: str) -> float:
        """Assess how complete and thorough the reasoning is."""
        if len(reasoning) < 50:
            return 0.2
        
        # Check for structured reasoning elements
        structure_indicators = [
            "first", "second", "additionally", "however", "therefore", 
            "because", "due to", "as a result", "furthermore"
        ]
        
        structure_count = sum(1 for indicator in structure_indicators if indicator in reasoning.lower())
        return min(structure_count / 3.0, 1.0)
    
    def _assess_evidence_specificity(self, result: Dict[str, Any]) -> float:
        """Assess how specific and evidence-based the judgment is."""
        strengths = result.get("strengths", [])
        weaknesses = result.get("weaknesses", [])
        
        # Check for specific examples vs vague statements
        specific_count = 0
        total_points = len(strengths) + len(weaknesses)
        
        if total_points == 0:
            return 0.0
        
        for item in strengths + weaknesses:
            if any(word in item.lower() for word in ["example", "specifically", "instance", "such as"]):
                specific_count += 1
        
        return specific_count / total_points
    
    def _assess_consistency(self, result: Dict[str, Any]) -> float:
        """Check consistency between score, strengths, and weaknesses."""
        score = result.get("score", 0)
        strengths_count = len(result.get("strengths", []))
        weaknesses_count = len(result.get("weaknesses", []))
        
        # High score should have more strengths than weaknesses
        if score > 0.7 and strengths_count < weaknesses_count:
            return 0.3
        
        # Low score should have more weaknesses than strengths  
        if score < 0.4 and strengths_count > weaknesses_count:
            return 0.3
        
        # Medium scores should be balanced
        if 0.4 <= score <= 0.7:
            balance = abs(strengths_count - weaknesses_count)
            return max(0.5, 1.0 - (balance * 0.2))
        
        return 0.8
    
    def _assess_confidence_calibration(self, result: Dict[str, Any]) -> float:
        """Assess if confidence level matches the evidence quality."""
        confidence = result.get("confidence", 0.5)
        evidence_quality = (
            len(result.get("strengths", [])) + len(result.get("weaknesses", []))
        ) / 6.0  # Normalized to 0-1
        
        # Well-calibrated confidence should match evidence quality
        calibration_error = abs(confidence - evidence_quality)
        return max(0.0, 1.0 - (calibration_error * 2))
    
    def _generate_quality_flags(self, quality_metrics: Dict[str, float]) -> List[str]:
        """Generate quality warning flags."""
        flags = []
        
        if quality_metrics["reasoning_completeness"] < 0.5:
            flags.append("Insufficient reasoning depth")
        
        if quality_metrics["evidence_specificity"] < 0.3:
            flags.append("Lacks specific evidence")
        
        if quality_metrics["consistency"] < 0.5:
            flags.append("Inconsistent score vs evidence")
        
        if quality_metrics["confidence_calibration"] < 0.4:
            flags.append("Poor confidence calibration")
        
        return flags
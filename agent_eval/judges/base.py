from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional
from agent_eval.models.wrapper_factory import ModelWrapperFactory
from agent_eval.judges.bias_detection import BiasDetector, MetaEvaluator
import json
import re
from datetime import datetime


class JudgeRegistry(ABCMeta):
    """
    Metaclass for automatically registering judge classes.
    """

    _registry = []
    _name_map = None
    _category_map = None

    def __init__(cls, name, bases, attrs):
        if name != "BaseJudge":
            JudgeRegistry._registry.append(cls)
        super().__init__(name, bases, attrs)

    @classmethod
    def _build_maps(cls):
        """Build name and category maps for efficient lookup."""
        cls._name_map = {}
        cls._category_map = {}
        for judge_cls in cls._registry:
            names = [judge_cls.__name__.replace("Judge", "").lower()]
            if hasattr(judge_cls, "aliases"):
                names += [alias.lower() for alias in judge_cls.aliases]
            for n in names:
                cls._name_map[n] = judge_cls

    @classmethod
    def get_by_name(cls, name: str):
        """
        Get a judge class by name, considering both the class name and aliases.
        """
        if cls._name_map is None:
            cls._build_maps()
        return cls._name_map.get(name.lower())

    @classmethod
    def get_by_category(cls, category):
        """
        Get all judge classes that support a specific category.
        """
        if cls._category_map is None:
            cls._build_maps()
        return cls._category_map.get(category, [])


class BaseJudge(ABC, metaclass=JudgeRegistry):
    """
    Enhanced base class for Chain-of-Thought evaluation judges.
    
    Implements research-backed techniques to reduce LLM-as-a-judge bias:
    - Chain-of-Thought reasoning for 15-30% accuracy improvement
    - Multi-perspective evaluation to reduce echo chamber effects
    - Systematic bias detection and correction
    - Meta-evaluation for judge quality assessment
    """

    def __init__(self, model, criteria: str, provider: str = None, enable_cot: bool = False):
        self.model = ModelWrapperFactory.wrap(model, provider)
        self.criteria = criteria
        self.bias_detector = BiasDetector()
        self.meta_evaluator = MetaEvaluator()
        self.enable_cot = enable_cot
        # Get prompt template - each judge must implement this
        self.prompt_template = self._get_prompt_template()

    @abstractmethod
    def judge(self, prompt, model_output, reference_output=None, **kwargs):
        """
        Judge a model output using specialized prompts with optional CoT enhancement.
        """
        pass
    
    @abstractmethod
    def _get_prompt_template(self):
        """
        Each judge must return its specific prompt template.
        """
        pass
    
    @abstractmethod
    def _get_judge_name(self):
        """
        Each judge must return its name for domain-specific fields.
        """
        pass

    def chain_of_thought_judge(self, prompt: str, model_output: str, reference_output: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Enhanced judgment using Chain-of-Thought reasoning to reduce bias.
        """
        # Step 1: Generate Chain-of-Thought evaluation
        cot_prompt = self._build_cot_prompt(prompt, model_output, reference_output)
        cot_judgment = self._generate_judgment(cot_prompt)
        
        # Step 2: Parse structured reasoning
        parsed_result = self._parse_cot_judgment(cot_judgment)
        
        # Step 3: Bias detection and correction
        bias_analysis = self.bias_detector.analyze(parsed_result, prompt, model_output)
        if bias_analysis.has_bias:
            parsed_result = self._correct_bias(parsed_result, bias_analysis)
        
        # Step 4: Meta-evaluation for judge quality
        meta_score = self.meta_evaluator.evaluate_judgment_quality(parsed_result)
        parsed_result["meta_evaluation"] = meta_score
        
        # Step 5: Add improvement suggestions
        parsed_result["improvement_suggestions"] = self._generate_improvement_suggestions(parsed_result)
        
        return parsed_result

    def _build_cot_prompt(self, prompt: str, model_output: str, reference_output: Optional[str] = None) -> str:
        """Build Chain-of-Thought evaluation prompt."""
        cot_template = """
You are an expert evaluator analyzing AI responses with systematic reasoning.

TASK: Evaluate the following AI response using step-by-step analysis.

USER PROMPT: {prompt}

AI RESPONSE: {model_output}

{reference_section}

EVALUATION CRITERIA: {criteria}

Use this structured reasoning process:

1. UNDERSTANDING ANALYSIS:
   - What is the core question/task?
   - What are the key requirements?
   - What would constitute a high-quality response?

2. CONTENT ANALYSIS:
   - What specific claims/information does the response contain?
   - Are these claims accurate and well-supported?
   - How comprehensive is the coverage?

3. QUALITY ASSESSMENT:
   - Relevance: How well does it address the question?
   - Accuracy: Are the facts correct?
   - Completeness: Does it cover all important aspects?
   - Clarity: Is it well-structured and understandable?

4. COMPARATIVE ANALYSIS:
   {comparison_instruction}

5. BIAS CHECK:
   - Does this evaluation avoid common biases?
   - Am I being fair and objective?
   - Are there alternative valid interpretations?

6. FINAL JUDGMENT:
   Provide your evaluation as JSON:
   {{
     "score": <float 0-1>,
     "reasoning": "Step-by-step explanation",
     "strengths": ["strength1", "strength2"],
     "weaknesses": ["weakness1", "weakness2"],
     "confidence": <float 0-1>,
     "alternative_perspectives": ["perspective1", "perspective2"]
   }}
"""
        
        reference_section = f"REFERENCE ANSWER: {reference_output}\n" if reference_output else ""
        comparison_instruction = "How does it compare to the reference answer?" if reference_output else "How does this response measure against best practices for this type of question?"
        
        return cot_template.format(
            prompt=prompt,
            model_output=model_output,
            reference_section=reference_section,
            criteria=self.criteria,
            comparison_instruction=comparison_instruction
        )

    def _generate_judgment(self, prompt: str) -> str:
        """
        Generate judgment based on the model's output.
        """
        try:
            return self.model.generate(prompt)
        except Exception as e:
            raise Exception(f"LLM judgment generation failed: {str(e)}")

    def _parse_cot_judgment(self, judgment_text: str) -> Dict[str, Any]:
        """Parse Chain-of-Thought judgment with enhanced structure."""
        try:
            match = re.search(r"\{.*\}", judgment_text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                return {
                    "score": parsed.get("score", 0),
                    "reasoning": parsed.get("reasoning", ""),
                    "strengths": parsed.get("strengths", []),
                    "weaknesses": parsed.get("weaknesses", []),
                    "confidence": parsed.get("confidence", 0.5),
                    "alternative_perspectives": parsed.get("alternative_perspectives", []),
                    "timestamp": datetime.now().isoformat(),
                    "raw_response": judgment_text
                }
            else:
                return self._fallback_parse(judgment_text)
        except Exception as e:
            return self._fallback_parse(judgment_text, error=str(e))

    def _fallback_parse(self, judgment_text: str, error: Optional[str] = None) -> Dict[str, Any]:
        """Fallback parsing for non-JSON responses."""
        score_match = re.search(r"\\bscore\\s*[:\-]?\\s*(\\d+(\\.\\d+)?)", judgment_text)
        score = float(score_match.group(1)) if score_match else 0.0
        
        return {
            "score": score,
            "reasoning": judgment_text.strip(),
            "strengths": [],
            "weaknesses": [],
            "confidence": 0.3,  # Lower confidence for fallback parsing
            "alternative_perspectives": [],
            "timestamp": datetime.now().isoformat(),
            "raw_response": judgment_text,
            "parsing_error": error
        }

    def _correct_bias(self, result: Dict[str, Any], bias_analysis: 'BiasAnalysis') -> Dict[str, Any]:
        """Apply bias correction to judgment results."""
        corrected_result = result.copy()
        
        # Adjust score based on detected bias
        if bias_analysis.bias_type == "leniency_bias":
            corrected_result["score"] = max(0.0, result["score"] - bias_analysis.adjustment)
        elif bias_analysis.bias_type == "severity_bias":
            corrected_result["score"] = min(1.0, result["score"] + bias_analysis.adjustment)
            
        # Add bias correction metadata
        corrected_result["bias_correction"] = {
            "original_score": result["score"],
            "corrected_score": corrected_result["score"],
            "bias_type": bias_analysis.bias_type,
            "confidence": bias_analysis.confidence
        }
        
        return corrected_result

    def _generate_improvement_suggestions(self, result: Dict[str, Any]) -> List[str]:
        """Generate specific, actionable improvement suggestions."""
        suggestions = []
        score = result.get("score", 0)
        weaknesses = result.get("weaknesses", [])
        
        # Score-based suggestions
        if score < 0.4:
            suggestions.append("Consider complete response restructuring - current approach has fundamental issues")
        elif score < 0.7:
            suggestions.append("Response needs significant improvements in accuracy and completeness")
        
        # Weakness-based suggestions
        for weakness in weaknesses:
            if "accuracy" in weakness.lower():
                suggestions.append("Add fact-checking step: Verify claims against reliable sources before responding")
            elif "relevance" in weakness.lower():
                suggestions.append("Improve query understanding: Analyze core question before generating response")
            elif "completeness" in weakness.lower():
                suggestions.append("Expand coverage: Address all aspects mentioned in the original question")
        
        return suggestions
    
    def evaluate(self, generated, reference=None, prompt=None, **kwargs):
        """
        Backward compatibility method - delegates to judge.
        Override in subclasses if needed for specific compatibility.
        """
        result = self.judge(prompt or "", generated, reference, **kwargs)
        return {
            'score': result['score'],
            'detailed_evaluation': result,
            'suggestion': result.get('improvement_suggestions', [''])[0] if result.get('improvement_suggestions') else "Consider reviewing and improving the response quality"
        }

    def _evaluate_with_specialized_prompt(self, prompt: str, model_output: str, reference_output: str = None) -> dict:
        """
        Evaluate using the judge's specialized prompt template.
        """
        # Format the specialized prompt
        judge_prompt = self.prompt_template.format(
            user_query=prompt,
            model_output=model_output,
            reference_output=reference_output or "No reference provided"
        )
        
        # Generate judgment using the specialized prompt
        judgment_text = self.model.generate(judge_prompt)
        
        # Parse the response
        parsed = self._parse_specialized_judgment(judgment_text)
        
        # Return objective result format
        return self._format_objective_result(parsed)
    
    def _evaluate_with_cot_enhancement(self, prompt: str, model_output: str, reference_output: str = None) -> dict:
        """
        Enhance specialized prompt evaluation with Chain-of-Thought analysis.
        """
        # Get base evaluation from specialized prompt
        base_result = self._evaluate_with_specialized_prompt(prompt, model_output, reference_output)
        
        # Enhance with CoT analysis
        cot_result = self.chain_of_thought_judge(prompt, model_output, reference_output)
        
        # Combine results - keep specialized prompt as primary, add CoT insights
        enhanced_result = base_result.copy()
        enhanced_result.update({
            "evaluation_method": "chain_of_thought_enhanced",
            "cot_analysis": {
                "bias_detection": cot_result.get("bias_correction"),
                "meta_evaluation": cot_result.get("meta_evaluation"),
                "alternative_perspectives": cot_result.get("alternative_perspectives", []),
                "enhanced_strengths": cot_result.get("strengths", []),
                "enhanced_weaknesses": cot_result.get("weaknesses", [])
            },
            "improvement_suggestions": cot_result.get("improvement_suggestions", []),
            "cot_enabled": True
        })
        
        return enhanced_result
    
    def _parse_specialized_judgment(self, judgment_text: str) -> dict:
        """
        Parse judgment from specialized prompt (expects JSON format).
        """
        try:
            # Extract JSON from response
            import json
            import re
            match = re.search(r'\{.*\}', judgment_text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                return {
                    "score": parsed.get("score", 0),
                    "reasoning": parsed.get("reasoning", ""),
                    "strengths": parsed.get("strengths", []),
                    "weaknesses": parsed.get("weaknesses", []),
                    "improvement_suggestions": parsed.get("improvement_suggestions", [])
                }
            else:
                # Fallback parsing
                return self._fallback_specialized_parse(judgment_text)
        except Exception as e:
            return self._fallback_specialized_parse(judgment_text, str(e))
    
    def _fallback_specialized_parse(self, judgment_text: str, error: str = None) -> dict:
        """
        Fallback parsing for non-JSON responses from specialized prompts.
        """
        # Try to extract score from text
        import re
        score_match = re.search(r'\bscore[:\s]*([1-5])(?:\.\d+)?', judgment_text.lower())
        score = int(score_match.group(1)) if score_match else 3
        
        return {
            "score": score,
            "reasoning": judgment_text.strip() or "Unable to parse detailed reasoning",
            "strengths": [],
            "weaknesses": [],
            "improvement_suggestions": [],
            "parsing_error": error
        }
    
    def _format_objective_result(self, parsed: dict) -> dict:
        """
        Format result in consistent, objective structure with no subjective interpretations.
        """
        score = parsed["score"] / 5.0  # Convert 1-5 scale to 0-1
        
        return {
            "score": score,
            "reasoning": parsed.get("reasoning", ""),
            "confidence": 0.85,  # High confidence with specialized prompts
            "evaluation_method": "specialized_prompt",
            "strengths": parsed.get("strengths", []),
            "weaknesses": parsed.get("weaknesses", []),
            "improvement_suggestions": parsed.get("improvement_suggestions", []),
            f"{self._get_judge_name()}_specific": self._get_objective_domain_fields(parsed, score),
            "timestamp": datetime.now().isoformat(),
            "cot_enabled": False
        }
    
    def _get_objective_domain_fields(self, parsed: dict, score: float) -> dict:
        """
        Get domain-specific objective fields. Override in subclasses.
        """
        return {}
    
    def _parse_judgment(self, judgment_text: str) -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        return self._parse_cot_judgment(judgment_text)

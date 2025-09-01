"""
Domain-Specific Judges Generator

Automatically generates custom judges based on detected domain and agent description.
This eliminates the need for manual judge creation.
"""

from typing import List, Dict, Any, Optional
from enum import Enum
import re

from agent_eval.domain.intelligence_engine import DomainType


class DomainJudge:
    """Represents a domain-specific judge."""
    
    def __init__(self, name: str, description: str, evaluation_prompt: str, 
                 expertise_areas: List[str], weight: float = 1.0):
        self.name = name
        self.description = description
        self.evaluation_prompt = evaluation_prompt
        self.expertise_areas = expertise_areas
        self.weight = weight
    
    def evaluate(self, prompt: str, model_output: str, reference_output: str = None,
                model=None, **kwargs) -> Dict[str, Any]:
        """Evaluate using the domain-specific judge criteria."""
        # This would use an LLM to evaluate based on the specialized prompt
        # For now, return a placeholder structure
        return {
            "score": 0.0,  # Will be filled by actual LLM evaluation
            "reasoning": "",
            "strengths": [],
            "weaknesses": [],
            "confidence": 0.8,
            "improvement_suggestions": [],
            "domain_specific_insights": {}
        }


class DomainJudgesGenerator:
    """Generates domain-specific judges automatically."""
    
    def __init__(self):
        self.domain_templates = {
            DomainType.FINANCE: self._get_finance_judge_templates(),
            DomainType.HEALTHCARE: self._get_healthcare_judge_templates(),
            DomainType.LEGAL: self._get_legal_judge_templates(),
            DomainType.GENERAL: self._get_general_judge_templates()
        }
    
    def generate_judges(self, domain: DomainType, agent_description: str,
                       prompt: str = None, model_output: str = None) -> List[DomainJudge]:
        """Generate custom judges for the specific domain and use case."""
        
        base_judges = self.domain_templates.get(domain, self.domain_templates[DomainType.GENERAL])
        
        # Customize based on agent description
        customized_judges = []
        for judge_template in base_judges:
            customized_judge = self._customize_judge(judge_template, agent_description, prompt, model_output)
            customized_judges.append(customized_judge)
        
        # Add specialized judges based on detected use case
        specialized_judges = self._generate_specialized_judges(domain, agent_description, prompt, model_output)
        customized_judges.extend(specialized_judges)
        
        return customized_judges
    
    def _get_finance_judge_templates(self) -> List[Dict[str, Any]]:
        """Finance domain judge templates."""
        return [
            {
                "name": "aml_compliance_judge",
                "description": "Evaluates AML compliance accuracy and regulatory adherence",
                "base_prompt": """
                You are an expert AML Compliance Officer evaluating an AI agent's response for AML compliance accuracy.
                
                Evaluate the response on:
                1. Accuracy of regulatory citations (BSA, PATRIOT Act, OFAC, FinCEN)
                2. Correctness of risk assessments and scoring
                3. Proper identification of suspicious activity indicators
                4. Appropriate compliance recommendations
                5. Accuracy of reporting requirements
                
                Provide a score from 0.0 to 1.0 and detailed reasoning.
                """,
                "expertise_areas": ["AML", "BSA", "PATRIOT Act", "OFAC", "FinCEN", "Risk Assessment"],
                "weight": 1.5
            },
            {
                "name": "regulatory_accuracy_judge", 
                "description": "Evaluates accuracy of financial regulatory information",
                "base_prompt": """
                You are a financial regulatory expert evaluating the accuracy of regulatory information.
                
                Evaluate the response on:
                1. Correct citation of relevant regulations
                2. Accurate interpretation of regulatory requirements
                3. Proper understanding of compliance obligations
                4. Current and up-to-date regulatory information
                5. Jurisdiction-specific accuracy
                
                Provide a score from 0.0 to 1.0 and detailed reasoning.
                """,
                "expertise_areas": ["Financial Regulations", "Compliance", "Legal Citations"],
                "weight": 1.4
            },
            {
                "name": "risk_assessment_judge",
                "description": "Evaluates quality of financial risk assessments",
                "base_prompt": """
                You are a risk management expert evaluating the quality of risk assessments.
                
                Evaluate the response on:
                1. Accuracy of risk scoring methodology
                2. Completeness of risk factor identification
                3. Appropriate risk categorization
                4. Quality of risk mitigation recommendations
                5. Understanding of risk interdependencies
                
                Provide a score from 0.0 to 1.0 and detailed reasoning.
                """,
                "expertise_areas": ["Risk Management", "Risk Scoring", "Financial Risk"],
                "weight": 1.3
            }
        ]
    
    def _get_healthcare_judge_templates(self) -> List[Dict[str, Any]]:
        """Healthcare domain judge templates."""
        return [
            {
                "name": "medical_accuracy_judge",
                "description": "Evaluates medical accuracy and clinical appropriateness",
                "base_prompt": """
                You are a medical expert evaluating the accuracy and appropriateness of medical information.
                
                Evaluate the response on:
                1. Medical accuracy and evidence-based information
                2. Appropriateness of clinical recommendations
                3. Safety considerations and contraindications
                4. Current medical standards and guidelines
                5. Risk-benefit analysis quality
                
                Provide a score from 0.0 to 1.0 and detailed reasoning.
                """,
                "expertise_areas": ["Medicine", "Clinical Practice", "Medical Safety"],
                "weight": 1.5
            },
            {
                "name": "patient_safety_judge",
                "description": "Evaluates patient safety and risk considerations",
                "base_prompt": """
                You are a patient safety expert evaluating safety considerations in medical advice.
                
                Evaluate the response on:
                1. Identification of safety risks
                2. Appropriate warnings and precautions
                3. Emergency situation recognition
                4. Proper medical disclaimer usage
                5. Risk communication effectiveness
                
                Provide a score from 0.0 to 1.0 and detailed reasoning.
                """,
                "expertise_areas": ["Patient Safety", "Risk Management", "Medical Ethics"],
                "weight": 1.4
            }
        ]
    
    def _get_legal_judge_templates(self) -> List[Dict[str, Any]]:
        """Legal domain judge templates."""
        return [
            {
                "name": "legal_accuracy_judge",
                "description": "Evaluates accuracy of legal information and advice",
                "base_prompt": """
                You are a legal expert evaluating the accuracy of legal information and advice.
                
                Evaluate the response on:
                1. Accuracy of legal information and citations
                2. Proper understanding of applicable law
                3. Jurisdiction-specific accuracy
                4. Current legal standards and precedents
                5. Appropriate legal reasoning
                
                Provide a score from 0.0 to 1.0 and detailed reasoning.
                """,
                "expertise_areas": ["Law", "Legal Research", "Case Analysis"],
                "weight": 1.5
            },
            {
                "name": "ethics_compliance_judge",
                "description": "Evaluates legal ethics and professional conduct compliance",
                "base_prompt": """
                You are a legal ethics expert evaluating compliance with professional conduct standards.
                
                Evaluate the response on:
                1. Adherence to legal ethics rules
                2. Proper attorney-client relationship boundaries
                3. Conflict of interest awareness
                4. Professional responsibility compliance
                5. Appropriate legal disclaimer usage
                
                Provide a score from 0.0 to 1.0 and detailed reasoning.
                """,
                "expertise_areas": ["Legal Ethics", "Professional Conduct", "Attorney Responsibility"],
                "weight": 1.3
            }
        ]
    
    def _get_general_judge_templates(self) -> List[Dict[str, Any]]:
        """General domain judge templates."""
        return [
            {
                "name": "helpfulness_judge",
                "description": "Evaluates overall helpfulness and usefulness of the response",
                "base_prompt": """
                You are evaluating the helpfulness and usefulness of an AI agent's response.
                
                Evaluate the response on:
                1. How well it addresses the user's question
                2. Practical value and actionability
                3. Clarity and understandability
                4. Completeness relative to the query
                5. Overall user satisfaction potential
                
                Provide a score from 0.0 to 1.0 and detailed reasoning.
                """,
                "expertise_areas": ["Communication", "User Experience", "Content Quality"],
                "weight": 1.0
            },
            {
                "name": "accuracy_judge",
                "description": "Evaluates factual accuracy and correctness",
                "base_prompt": """
                You are evaluating the factual accuracy and correctness of an AI agent's response.
                
                Evaluate the response on:
                1. Factual correctness of stated information
                2. Logical consistency and reasoning
                3. Avoidance of misinformation
                4. Appropriate uncertainty expression
                5. Evidence-based claims
                
                Provide a score from 0.0 to 1.0 and detailed reasoning.
                """,
                "expertise_areas": ["Fact Checking", "Logical Reasoning", "Information Verification"],
                "weight": 1.2
            }
        ]
    
    def _customize_judge(self, judge_template: Dict[str, Any], agent_description: str,
                        prompt: str = None, model_output: str = None) -> DomainJudge:
        """Customize a judge template for specific agent use case."""
        
        customized_prompt = judge_template["base_prompt"]
        
        # Add specific customizations based on agent description
        if agent_description and "AML" in agent_description.upper():
            if "compliance" in judge_template["name"]:
                customized_prompt += "\n\nPay special attention to AML-specific requirements including transaction monitoring, customer due diligence, and suspicious activity reporting."
            elif "risk" in judge_template["name"]:
                customized_prompt += "\n\nFocus on money laundering risk factors, customer risk profiles, and transaction risk assessment accuracy."
        
        return DomainJudge(
            name=judge_template["name"],
            description=judge_template["description"], 
            evaluation_prompt=customized_prompt,
            expertise_areas=judge_template["expertise_areas"],
            weight=judge_template["weight"]
        )
    
    def _generate_specialized_judges(self, domain: DomainType, agent_description: str,
                                   prompt: str = None, model_output: str = None) -> List[DomainJudge]:
        """Generate specialized judges based on specific use case detection."""
        specialized = []
        
        if domain == DomainType.FINANCE and agent_description:
            # AML-specific judges
            if any(keyword in agent_description.upper() for keyword in ["AML", "MONEY LAUNDERING", "SUSPICIOUS"]):
                specialized.extend([
                    DomainJudge(
                        name="structuring_detection_judge",
                        description="Evaluates ability to detect transaction structuring patterns",
                        evaluation_prompt="""
                        You are an AML expert specializing in structuring detection, evaluating an AI agent's ability to identify structuring patterns.
                        
                        Evaluate the response on:
                        1. Correct identification of structuring patterns
                        2. Understanding of threshold avoidance tactics
                        3. Recognition of smurfing activities
                        4. Proper risk scoring for structuring
                        5. Appropriate regulatory response recommendations
                        
                        Provide a score from 0.0 to 1.0 and detailed reasoning.
                        """,
                        expertise_areas=["Structuring Detection", "Transaction Monitoring", "AML Patterns"],
                        weight=1.3
                    ),
                    DomainJudge(
                        name="sanctions_screening_judge", 
                        description="Evaluates sanctions screening accuracy and completeness",
                        evaluation_prompt="""
                        You are an OFAC sanctions expert evaluating the accuracy of sanctions screening processes.
                        
                        Evaluate the response on:
                        1. Proper OFAC SDN list screening procedures
                        2. Correct PEP identification and handling
                        3. Watch list management accuracy
                        4. Sanctions violation risk assessment
                        5. Appropriate escalation procedures
                        
                        Provide a score from 0.0 to 1.0 and detailed reasoning.
                        """,
                        expertise_areas=["OFAC Sanctions", "PEP Screening", "Watch Lists"],
                        weight=1.4
                    ),
                    DomainJudge(
                        name="typology_recognition_judge",
                        description="Evaluates recognition of money laundering typologies",
                        evaluation_prompt="""
                        You are an AML typology expert evaluating recognition of money laundering patterns and methods.
                        
                        Evaluate the response on:
                        1. Correct identification of ML typologies (layering, integration, etc.)
                        2. Understanding of trade-based money laundering
                        3. Recognition of digital/crypto ML methods
                        4. Geographic risk pattern identification
                        5. Sector-specific ML method awareness
                        
                        Provide a score from 0.0 to 1.0 and detailed reasoning.
                        """,
                        expertise_areas=["ML Typologies", "TBML", "Geographic Risk"],
                        weight=1.2
                    )
                ])
        
        return specialized
    
    def get_judge_weights(self, judges: List[DomainJudge]) -> Dict[str, float]:
        """Get normalized weights for all judges."""
        total_weight = sum(j.weight for j in judges)
        return {j.name: j.weight / total_weight for j in judges}
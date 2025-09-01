"""
Domain-Specific Metrics Generator

Automatically generates custom metrics based on detected domain and agent description.
This eliminates the need for manual metric creation.
"""

from typing import List, Dict, Any, Optional
from enum import Enum
import re

from agent_eval.domain.intelligence_engine import DomainType


class DomainMetric:
    """Represents a domain-specific metric."""
    
    def __init__(self, name: str, description: str, scoring_criteria: str, 
                 weight: float = 1.0, threshold: float = 0.7):
        self.name = name
        self.description = description
        self.scoring_criteria = scoring_criteria
        self.weight = weight
        self.threshold = threshold
    
    def evaluate(self, prompt: str, model_output: str, reference_output: str = None) -> Dict[str, Any]:
        """Evaluate the metric using LLM-based scoring."""
        # This would use an LLM to score based on the criteria
        # For now, return a placeholder structure
        return {
            "score": 0.0,  # Will be filled by actual LLM evaluation
            "reasoning": "",
            "criteria": self.scoring_criteria,
            "suggestions": []
        }


class DomainMetricsGenerator:
    """Generates domain-specific metrics automatically."""
    
    def __init__(self):
        self.domain_templates = {
            DomainType.FINANCE: self._get_finance_metric_templates(),
            DomainType.HEALTHCARE: self._get_healthcare_metric_templates(),
            DomainType.LEGAL: self._get_legal_metric_templates(),
            DomainType.GENERAL: self._get_general_metric_templates()
        }
    
    def generate_metrics(self, domain: DomainType, agent_description: str, 
                        prompt: str = None, model_output: str = None) -> List[DomainMetric]:
        """Generate custom metrics for the specific domain and use case."""
        
        base_metrics = self.domain_templates.get(domain, self.domain_templates[DomainType.GENERAL])
        
        # Customize based on agent description
        customized_metrics = []
        for metric_template in base_metrics:
            customized_metric = self._customize_metric(metric_template, agent_description, prompt, model_output)
            customized_metrics.append(customized_metric)
        
        # Add specialized metrics based on detected use case
        specialized_metrics = self._generate_specialized_metrics(domain, agent_description, prompt, model_output)
        customized_metrics.extend(specialized_metrics)
        
        return customized_metrics
    
    def _get_finance_metric_templates(self) -> List[Dict[str, Any]]:
        """Finance domain metric templates."""
        return [
            {
                "name": "regulatory_compliance",
                "description": "Accuracy of regulatory citations and compliance requirements",
                "base_criteria": "Evaluate the accuracy of regulatory citations, compliance requirements, and adherence to financial regulations",
                "weight": 1.5,
                "threshold": 0.8
            },
            {
                "name": "risk_assessment_accuracy",
                "description": "Quality of risk scoring and risk factor identification",
                "base_criteria": "Assess the accuracy of risk scores, risk factor identification, and risk categorization",
                "weight": 1.3,
                "threshold": 0.7
            },
            {
                "name": "aml_pattern_recognition",
                "description": "Ability to identify money laundering patterns and typologies",
                "base_criteria": "Evaluate detection of ML patterns like structuring, layering, integration, and suspicious activity indicators",
                "weight": 1.4,
                "threshold": 0.75
            },
            {
                "name": "reporting_completeness",
                "description": "Completeness and accuracy of required regulatory reports",
                "base_criteria": "Assess whether all required reporting elements are included and correctly formatted",
                "weight": 1.2,
                "threshold": 0.8
            },
            {
                "name": "sanctions_screening_accuracy",
                "description": "Accuracy of sanctions screening and watch list checks",
                "base_criteria": "Evaluate proper identification of sanctioned entities and PEP connections",
                "weight": 1.3,
                "threshold": 0.85
            }
        ]
    
    def _get_healthcare_metric_templates(self) -> List[Dict[str, Any]]:
        """Healthcare domain metric templates."""
        return [
            {
                "name": "medical_accuracy",
                "description": "Accuracy of medical information and clinical recommendations",
                "base_criteria": "Evaluate medical accuracy, appropriateness of recommendations, and clinical soundness",
                "weight": 1.5,
                "threshold": 0.9
            },
            {
                "name": "safety_compliance",
                "description": "Adherence to medical safety guidelines and protocols",
                "base_criteria": "Assess compliance with medical safety standards and risk mitigation",
                "weight": 1.4,
                "threshold": 0.85
            },
            {
                "name": "patient_privacy",
                "description": "Protection of patient privacy and HIPAA compliance",
                "base_criteria": "Evaluate adherence to patient privacy requirements and data protection",
                "weight": 1.3,
                "threshold": 0.9
            }
        ]
    
    def _get_legal_metric_templates(self) -> List[Dict[str, Any]]:
        """Legal domain metric templates."""
        return [
            {
                "name": "legal_accuracy",
                "description": "Accuracy of legal information and citations",
                "base_criteria": "Evaluate accuracy of legal information, case citations, and statutory references",
                "weight": 1.5,
                "threshold": 0.85
            },
            {
                "name": "jurisdictional_relevance",
                "description": "Relevance to specified jurisdiction and applicable law",
                "base_criteria": "Assess whether legal advice is relevant to the correct jurisdiction and legal framework",
                "weight": 1.3,
                "threshold": 0.8
            },
            {
                "name": "ethical_compliance",
                "description": "Adherence to legal ethics and professional standards",
                "base_criteria": "Evaluate compliance with legal ethics rules and professional conduct standards",
                "weight": 1.4,
                "threshold": 0.9
            }
        ]
    
    def _get_general_metric_templates(self) -> List[Dict[str, Any]]:
        """General domain metric templates."""
        return [
            {
                "name": "response_quality",
                "description": "Overall quality and usefulness of the response",
                "base_criteria": "Evaluate overall response quality, clarity, and usefulness",
                "weight": 1.0,
                "threshold": 0.7
            },
            {
                "name": "instruction_following",
                "description": "How well the agent followed the given instructions",
                "base_criteria": "Assess adherence to provided instructions and requirements",
                "weight": 1.2,
                "threshold": 0.75
            },
            {
                "name": "completeness",
                "description": "Completeness of the response relative to the query",
                "base_criteria": "Evaluate whether the response fully addresses all aspects of the query",
                "weight": 1.1,
                "threshold": 0.7
            }
        ]
    
    def _customize_metric(self, metric_template: Dict[str, Any], agent_description: str, 
                         prompt: str = None, model_output: str = None) -> DomainMetric:
        """Customize a metric template for specific agent use case."""
        
        # Analyze the agent description for specific use case
        customized_criteria = metric_template["base_criteria"]
        
        # Add specific customizations based on agent description
        if agent_description and "AML" in agent_description.upper():
            if "regulatory" in metric_template["name"]:
                customized_criteria += ". Focus on BSA, PATRIOT Act, OFAC, and FinCEN requirements."
            elif "risk" in metric_template["name"]:
                customized_criteria += ". Emphasize customer due diligence and transaction risk factors."
        
        return DomainMetric(
            name=metric_template["name"],
            description=metric_template["description"],
            scoring_criteria=customized_criteria,
            weight=metric_template["weight"],
            threshold=metric_template["threshold"]
        )
    
    def _generate_specialized_metrics(self, domain: DomainType, agent_description: str,
                                    prompt: str = None, model_output: str = None) -> List[DomainMetric]:
        """Generate specialized metrics based on specific use case detection."""
        specialized = []
        
        if domain == DomainType.FINANCE and agent_description:
            # AML-specific metrics
            if any(keyword in agent_description.upper() for keyword in ["AML", "MONEY LAUNDERING", "SUSPICIOUS"]):
                specialized.extend([
                    DomainMetric(
                        name="structuring_detection",
                        description="Ability to detect structuring patterns in transactions",
                        scoring_criteria="Evaluate detection of transaction structuring patterns and threshold avoidance tactics",
                        weight=1.3,
                        threshold=0.8
                    ),
                    DomainMetric(
                        name="pep_identification",
                        description="Accuracy in identifying Politically Exposed Persons",
                        scoring_criteria="Assess correct identification of PEPs and associated risk factors",
                        weight=1.2,
                        threshold=0.85
                    ),
                    DomainMetric(
                        name="geographic_risk_assessment",
                        description="Evaluation of geographic and jurisdictional risks",
                        scoring_criteria="Evaluate proper assessment of geographic risks and high-risk jurisdictions",
                        weight=1.1,
                        threshold=0.75
                    )
                ])
        
        return specialized
    
    def get_metric_weights(self, metrics: List[DomainMetric]) -> Dict[str, float]:
        """Get normalized weights for all metrics."""
        total_weight = sum(m.weight for m in metrics)
        return {m.name: m.weight / total_weight for m in metrics}
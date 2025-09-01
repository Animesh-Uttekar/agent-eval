"""
Test helpers and mock models for judge testing.
"""

import json
import re


class MockModel:
    """Mock LLM model for testing judges."""
    
    def __init__(self, mock_responses=None):
        self.mock_responses = mock_responses or {}
        self.call_count = 0
        self.last_prompt = None
    
    def generate(self, prompt):
        """Generate mock response based on prompt keywords."""
        self.last_prompt = prompt
        self.call_count += 1
        
        # Check for specific test scenarios in mock_responses
        for keyword, response in self.mock_responses.items():
            if keyword.lower() in prompt.lower():
                return response
        
        # Default responses based on judge type
        if "coherence" in prompt.lower():
            return self._mock_coherence_response()
        elif "completeness" in prompt.lower():
            return self._mock_completeness_response()
        elif "bias" in prompt.lower():
            return self._mock_bias_response()
        elif "healthcare" in prompt.lower() or "medical" in prompt.lower():
            return self._mock_healthcare_response()
        elif "technical" in prompt.lower() or "code" in prompt.lower():
            return self._mock_technical_response()
        elif "legal" in prompt.lower():
            return self._mock_legal_response()
        elif "empathy" in prompt.lower() or "emotional" in prompt.lower():
            return self._mock_empathy_response()
        elif "educational" in prompt.lower() or "learning" in prompt.lower():
            return self._mock_educational_response()
        elif "aml" in prompt.lower() or "compliance" in prompt.lower():
            return self._mock_aml_response()
        elif "financial" in prompt.lower():
            return self._mock_financial_response()
        else:
            return self._mock_default_response()
    
    def _mock_coherence_response(self):
        return '''
        {
          "score": 4,
          "reasoning": "The response demonstrates good logical flow with clear progression of ideas. Structure is well-organized with appropriate transitions between topics.",
          "strengths": ["Clear logical progression", "Well-organized structure", "Good transitions"],
          "weaknesses": ["Minor inconsistency in tone", "Could use better signposting"],
          "improvement_suggestions": ["Add more explicit signposting", "Ensure consistent tone throughout"]
        }
        '''
    
    def _mock_completeness_response(self):
        return '''
        {
          "score": 4,
          "reasoning": "The response addresses most aspects of the query with good detail, though some areas could be expanded.",
          "strengths": ["Covers main query points", "Provides good detail", "Includes relevant context"],
          "weaknesses": ["Missing some edge cases", "Could provide more actionable steps"],
          "improvement_suggestions": ["Include more specific examples", "Add actionable next steps"]
        }
        '''
    
    def _mock_bias_response(self):
        return '''
        {
          "score": 5,
          "reasoning": "The response demonstrates excellent fairness with inclusive language and balanced perspectives.",
          "strengths": ["Inclusive language", "Balanced representation", "Cultural sensitivity"],
          "weaknesses": [],
          "improvement_suggestions": ["Continue maintaining inclusive approach"]
        }
        '''
    
    def _mock_healthcare_response(self):
        return '''
        {
          "score": 3,
          "reasoning": "The response provides generally accurate medical information but lacks proper disclaimers and safety warnings.",
          "strengths": ["Accurate medical terminology", "Relevant clinical information"],
          "weaknesses": ["Missing medical disclaimers", "Insufficient safety warnings"],
          "improvement_suggestions": ["Add clear medical disclaimers", "Include more safety warnings"]
        }
        '''
    
    def _mock_technical_response(self):
        return '''
        {
          "score": 4,
          "reasoning": "The response demonstrates strong technical accuracy with good code examples and security considerations.",
          "strengths": ["Accurate technical concepts", "Good code examples", "Security awareness"],
          "weaknesses": ["Minor performance optimization missed"],
          "improvement_suggestions": ["Add performance optimization notes", "Include more error handling examples"]
        }
        '''
    
    def _mock_legal_response(self):
        return '''
        {
          "score": 3,
          "reasoning": "The response provides legal information but lacks proper citations and adequate disclaimers.",
          "strengths": ["Basic legal concepts explained", "Some relevant legal principles"],
          "weaknesses": ["Missing proper legal citations", "Inadequate legal disclaimers"],
          "improvement_suggestions": ["Add proper legal citations", "Include comprehensive legal disclaimers"]
        }
        '''
    
    def _mock_empathy_response(self):
        return '''
        {
          "score": 5,
          "reasoning": "The response demonstrates excellent empathy with supportive language and emotional awareness.",
          "strengths": ["Warm, supportive tone", "Good emotional validation", "Appropriate empathy"],
          "weaknesses": [],
          "improvement_suggestions": ["Continue empathetic approach"]
        }
        '''
    
    def _mock_educational_response(self):
        return '''
        {
          "score": 4,
          "reasoning": "The response provides good educational value with clear explanations and practical examples.",
          "strengths": ["Clear explanations", "Good examples", "Engaging presentation"],
          "weaknesses": ["Could use more interactive elements", "Missing practice opportunities"],
          "improvement_suggestions": ["Add interactive questions", "Include practice exercises"]
        }
        '''
    
    def _mock_aml_response(self):
        return '''
        {
          "score": 4,
          "reasoning": "The response demonstrates strong AML compliance knowledge with accurate regulatory references.",
          "strengths": ["Accurate regulatory citations", "Good risk identification", "Proper compliance procedures"],
          "weaknesses": ["Could include more specific thresholds"],
          "improvement_suggestions": ["Include specific dollar thresholds", "Add more regulatory examples"]
        }
        '''
    
    def _mock_financial_response(self):
        return '''
        {
          "score": 4,
          "reasoning": "The response demonstrates good financial expertise with appropriate terminology and practical insights.",
          "strengths": ["Correct financial terminology", "Good practical application", "Professional communication"],
          "weaknesses": ["Could include more quantitative analysis"],
          "improvement_suggestions": ["Add more quantitative analysis", "Include market context"]
        }
        '''
    
    def _mock_default_response(self):
        return '''
        {
          "score": 3,
          "reasoning": "The response provides a reasonable evaluation with some strengths and areas for improvement.",
          "strengths": ["Generally accurate information", "Appropriate tone"],
          "weaknesses": ["Could be more comprehensive", "Missing some details"],
          "improvement_suggestions": ["Add more specific examples", "Provide more detailed analysis"]
        }
        '''


class MockOpenAIModel:
    """Mock OpenAI-style model for testing."""
    
    def __init__(self, mock_responses=None):
        self.chat = MockChat(mock_responses)


class MockChat:
    """Mock chat interface."""
    
    def __init__(self, mock_responses=None):
        self.completions = MockCompletions(mock_responses)


class MockCompletions:
    """Mock completions interface."""
    
    def __init__(self, mock_responses=None):
        self.mock_model = MockModel(mock_responses)
    
    def create(self, model, messages, temperature=0.1, max_tokens=800):
        # Extract the user message content
        user_message = None
        for message in messages:
            if message.get("role") == "user":
                user_message = message.get("content", "")
                break
        
        response_content = self.mock_model.generate(user_message or "")
        
        return MockResponse(response_content)


class MockResponse:
    """Mock response object."""
    
    def __init__(self, content):
        self.choices = [MockChoice(content)]


class MockChoice:
    """Mock choice object."""
    
    def __init__(self, content):
        self.message = MockMessage(content)


class MockMessage:
    """Mock message object."""
    
    def __init__(self, content):
        self.content = content
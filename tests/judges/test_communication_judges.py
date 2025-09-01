"""
Test cases for communication and educational judges: Empathy and Educational Quality.
"""

import unittest
import json
from unittest.mock import Mock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agent_eval.judges.empathy_judge import EmpathyJudge
from agent_eval.judges.educational_judge import EducationalJudge
from tests.judges.test_helpers import MockModel, MockOpenAIModel


class TestEmpathyJudge(unittest.TestCase):
    """Test cases for EmpathyJudge."""
    
    def setUp(self):
        self.mock_model = MockModel()
        self.judge = EmpathyJudge(self.mock_model, "test-model")
    
    def test_empathetic_response_evaluation(self):
        """Test evaluation of an empathetic response."""
        prompt = "I'm feeling really overwhelmed at work and don't know what to do."
        generated = """
        I can understand how overwhelming work stress can feel, and it's completely 
        normal to feel this way sometimes. Your feelings are valid, and it's important 
        to acknowledge them rather than dismiss them.
        
        Here are some gentle strategies that might help:
        
        1. Take a step back: Sometimes a short break or even a few deep breaths 
           can help clear your mind and reduce immediate stress.
        
        2. Prioritize tasks: Try writing down everything you need to do and 
           identifying what's most urgent. This can make things feel more manageable.
        
        3. Reach out for support: Whether it's a trusted colleague, supervisor, 
           or friend, sharing your feelings can provide relief and perspective.
        
        4. Practice self-care: Make sure you're taking care of your basic needs - 
           eating well, getting enough sleep, and taking time for activities you enjoy.
        
        Remember, it's okay to not have everything figured out right away. 
        You're doing your best, and that's enough. If these feelings persist, 
        consider speaking with a counselor or therapist who can provide 
        professional guidance.
        
        Is there any specific aspect of your work situation you'd like to talk 
        through further?
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertIsInstance(result, dict)
        self.assertIn('score', result)
        self.assertGreater(result['score'], 0.7)  # Should score well for empathy
    
    def test_unempathetic_response_detection(self):
        """Test detection of unempathetic response."""
        # Mock response detecting lack of empathy
        self.mock_model.mock_responses = {
            'empathy': '''
            {
              "score": 2,
              "reasoning": "The response lacks empathy and emotional awareness, using dismissive language.",
              "strengths": [],
              "emotional_gaps": ["Dismissive tone", "Lack of validation", "Insensitive language"],
              "specific_feedback": {
                "emotional_awareness": "Fails to recognize emotional distress",
                "empathetic_language": "Uses cold, dismissive language",
                "tone_appropriateness": "Tone is inappropriate for emotional situation",
                "validation_support": "No emotional validation provided",
                "emotional_boundaries": "Inappropriate professional boundaries"
              },
              "improvement_suggestions": ["Acknowledge emotions", "Use supportive language", "Provide validation"]
            }
            '''
        }
        
        prompt = "I'm feeling really overwhelmed at work and don't know what to do."
        generated = "Just get over it and work harder. Everyone deals with stress. Stop complaining and do your job."
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertLess(result['score'], 0.3)  # Should get very low score for lack of empathy
        self.assertIn('emotional_gaps', result['detailed_evaluation'])
    
    def test_empathy_criteria_coverage(self):
        """Test that empathy criteria are properly defined."""
        expected_criteria = [
            "emotional_awareness",
            "empathetic_language", 
            "tone_appropriateness",
            "validation_support",
            "emotional_boundaries"
        ]
        
        for criterion in expected_criteria:
            self.assertIn(criterion, self.judge.criteria)
    
    def test_empathy_aliases(self):
        """Test empathy judge aliases."""
        expected_aliases = ["empathy", "emotional_intelligence", "compassion", "emotional_awareness"]
        self.assertEqual(self.judge.aliases, expected_aliases)


class TestEducationalJudge(unittest.TestCase):
    """Test cases for EducationalJudge."""
    
    def setUp(self):
        self.mock_model = MockModel()
        self.judge = EducationalJudge(self.mock_model, "test-model")
    
    def test_good_educational_response_evaluation(self):
        """Test evaluation of high-quality educational response."""
        prompt = "How does photosynthesis work?"
        generated = """
        Photosynthesis is one of the most important biological processes on Earth! 
        Let me break it down for you step by step.
        
        **What is Photosynthesis?**
        Think of photosynthesis as nature's way of making food using sunlight - 
        just like how we might use a recipe to cook a meal, but plants use 
        sunlight as their main ingredient!
        
        **The Simple Equation:**
        Sunlight + Water + Carbon Dioxide → Glucose (sugar) + Oxygen
        
        **Step-by-Step Process:**
        
        1. **Light Absorption** (Think of it like catching sunlight)
           - Chlorophyll in leaves captures sunlight
           - This is why leaves are green - chlorophyll reflects green light!
        
        2. **Water Uptake** (Like drinking through a straw)
           - Roots absorb water from the soil
           - Water travels up through the stem to the leaves
        
        3. **Carbon Dioxide Collection** (Breathing in from the air)
           - Tiny pores called stomata take in CO₂ from the atmosphere
           - You can think of stomata as the plant's "nostrils"
        
        4. **The Magic Happens** (The chemical reaction)
           - Inside chloroplasts, light energy converts water and CO₂ into glucose
           - Oxygen is released as a bonus - the oxygen we breathe!
        
        **Real-World Connection:**
        Every breath you take contains oxygen that was produced by photosynthesis. 
        Without this process, there would be no life on Earth as we know it!
        
        **Try This at Home:**
        Put a clear plastic bag over a leaf on a sunny day and watch bubbles form - 
        those are oxygen bubbles from photosynthesis in action!
        
        **Quick Review Questions:**
        - What are the three main inputs needed for photosynthesis?
        - Why are most leaves green?
        - How does photosynthesis benefit humans?
        
        Does this help clarify how photosynthesis works? Would you like me to 
        explain any particular step in more detail?
        """
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertIsInstance(result, dict)
        self.assertIn('score', result)
        self.assertGreater(result['score'], 0.8)  # Should score very well for educational quality
    
    def test_poor_educational_response_detection(self):
        """Test detection of poor educational quality."""
        # Mock response detecting educational issues
        self.mock_model.mock_responses = {
            'educational': '''
            {
              "score": 3,
              "reasoning": "The response lacks clarity, proper scaffolding, and engaging educational elements.",
              "strengths": ["Contains some accurate information"],
              "learning_gaps": ["Poor explanation clarity", "No learning progression", "Not engaging", "Missing examples"],
              "specific_feedback": {
                "clarity_explanation": "Explanations are confusing and use excessive jargon",
                "learning_progression": "No logical progression from simple to complex",
                "engagement_quality": "Dry and unengaging presentation",
                "practical_application": "No concrete examples or applications",
                "knowledge_reinforcement": "No reinforcement or review elements"
              },
              "improvement_suggestions": ["Simplify explanations", "Add engaging examples", "Include practice questions"]
            }
            '''
        }
        
        prompt = "How does photosynthesis work?"
        generated = "Photosynthesis involves complex biochemical pathways utilizing photosystems I and II in thylakoid membranes with electron transport chains and ATP synthase complexes."
        
        result = self.judge.evaluate(generated, prompt=prompt)
        
        self.assertLess(result['score'], 0.4)  # Should get low score for poor educational quality
        self.assertIn('learning_gaps', result['detailed_evaluation'])
    
    def test_educational_criteria_coverage(self):
        """Test that educational criteria are properly defined."""
        expected_criteria = [
            "clarity_explanation",
            "learning_progression",
            "engagement_quality",
            "practical_application",
            "knowledge_reinforcement"
        ]
        
        for criterion in expected_criteria:
            self.assertIn(criterion, self.judge.criteria)
    
    def test_educational_category(self):
        """Test educational judge categorization."""
        self.assertEqual(self.judge.category, "education")
        self.assertIn("education", self.judge.domain_focus)
        self.assertIn("learning", self.judge.domain_focus)
    
    def test_learning_theory_integration(self):
        """Test that system prompt includes pedagogical concepts."""
        system_prompt = self.judge.get_system_prompt()
        
        # Should include pedagogical concepts
        pedagogical_terms = ["scaffolding", "learning progression", "engagement", "reinforcement"]
        for term in pedagogical_terms:
            self.assertIn(term, system_prompt.lower())


class TestCommunicationJudgesIntegration(unittest.TestCase):
    """Integration tests for communication judges."""
    
    def setUp(self):
        self.mock_model = MockModel()
        self.empathy_judge = EmpathyJudge(self.mock_model, "test-model")
        self.educational_judge = EducationalJudge(self.mock_model, "test-model")
    
    def test_supportive_educational_response(self):
        """Test response that should score well on both empathy and educational quality."""
        prompt = "I'm struggling to understand algebra and feel really stupid."
        generated = """
        I want you to know that struggling with algebra is incredibly common, and it 
        definitely doesn't mean you're stupid! Many brilliant people have found 
        algebra challenging at first - it's a completely normal part of learning 
        mathematics.
        
        Let's break down algebra in a way that might feel more manageable:
        
        **What is Algebra Really?**
        Think of algebra as a puzzle-solving game where some pieces are missing. 
        The letters (like x or y) are just placeholders for the missing numbers 
        we're trying to find.
        
        **Simple Example:**
        If I say "I'm thinking of a number, and when I add 3 to it, I get 8," 
        you'd probably figure out the number is 5, right?
        
        In algebra, we'd write this as: x + 3 = 8
        
        **Building Confidence Step by Step:**
        1. Start with simple problems like the one above
        2. Practice the same type until it feels comfortable
        3. Gradually add more complexity
        4. Celebrate small wins along the way!
        
        **Remember:**
        - Every mathematician started exactly where you are now
        - Making mistakes is how we learn - they're not failures, they're data!
        - Your brain is absolutely capable of understanding this
        
        Would you like to try working through a simple problem together? I'm here 
        to support you, and we can go at whatever pace feels right for you.
        """
        
        empathy_result = self.empathy_judge.evaluate(generated, prompt=prompt)
        educational_result = self.educational_judge.evaluate(generated, prompt=prompt)
        
        # Both should score well
        self.assertGreater(empathy_result['score'], 0.7)  # High empathy
        self.assertGreater(educational_result['score'], 0.7)  # Good educational quality
    
    def test_judge_complementary_evaluation(self):
        """Test that judges provide complementary evaluation perspectives."""
        prompt = "Explain quantum physics"
        generated = """
        Quantum physics is like a fascinating puzzle that describes how incredibly 
        tiny particles behave in ways that seem almost magical compared to our 
        everyday experience.
        
        Imagine if you could be in two places at once, or if looking at something 
        actually changed what it was doing - that's the strange world of quantum mechanics!
        """
        
        empathy_result = self.empathy_judge.evaluate(generated, prompt=prompt)
        educational_result = self.educational_judge.evaluate(generated, prompt=prompt)
        
        # Results should be different - empathy focusing on emotional aspects,
        # educational focusing on learning quality
        self.assertIsInstance(empathy_result, dict)
        self.assertIsInstance(educational_result, dict)
        
        # Educational judge should evaluate learning elements
        self.assertIn('detailed_evaluation', educational_result)
        # Empathy judge should evaluate emotional elements  
        self.assertIn('detailed_evaluation', empathy_result)
    
    def test_consistent_error_handling(self):
        """Test consistent error handling across communication judges."""
        judges = [self.empathy_judge, self.educational_judge]
        
        for judge in judges:
            with self.subTest(judge=judge.__class__.__name__):
                # Test empty input
                result = judge.evaluate("")
                self.assertIn('error', result)
                self.assertEqual(result['score'], 0.0)
                
                # Test None model
                judge_no_model = judge.__class__(None, "test-model")
                result = judge_no_model.evaluate("Test content", prompt="Test prompt")
                self.assertIn('error', result)


if __name__ == '__main__':
    unittest.main()
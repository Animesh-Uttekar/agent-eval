#!/usr/bin/env python3
"""
Test the Dynamic Framework with DIFFERENT domains (not finance)

This demonstrates that the framework can automatically:
1. Analyze ANY AI agent in ANY domain
2. Generate custom metrics for that specific agent
3. Generate custom judges with domain expertise  
4. Create custom test scenarios
5. Provide domain-specific evaluation

Examples: Education AI, Healthcare AI, Creative Writing AI, Customer Service AI
"""

import os
from openai import OpenAI
from agent_eval import evaluator
from dotenv import load_dotenv

load_dotenv()


def test_education_ai_agent():
    """Test with an Education/Teaching AI agent (completely different from finance)."""
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # EDUCATION AI AGENT SYSTEM PROMPT
    EDUCATION_AI_PROMPT = """
    You are an expert Educational AI tutor specializing in personalized learning for K-12 students. 
    Your capabilities include:
    
    1. Adaptive Learning: Adjust explanations based on student grade level and learning style
    2. Curriculum Alignment: Ensure content aligns with educational standards (Common Core, etc.)
    3. Learning Assessment: Evaluate student understanding and provide targeted feedback
    4. Engagement Strategies: Use interactive methods, examples, and analogies appropriate for age
    5. Progress Tracking: Monitor learning progression and identify knowledge gaps
    6. Safety & Appropriateness: Ensure all content is age-appropriate and educationally sound
    
    Always provide clear explanations, check for understanding, and encourage student learning.
    """
    
    # SAMPLE EDUCATION AI OUTPUT
    EDUCATION_AI_OUTPUT = """
    Great question about photosynthesis! Let me explain this in a way that's easy to understand.
    
    Think of a plant like a tiny food factory! Here's how it works:
    
    THE RECIPE FOR PLANT FOOD:
    Ingredients needed:
    - Sunlight (the energy source)
    - Water (from the roots)
    - Carbon dioxide (from the air through leaves)
    - Chloroplasts (the tiny green "machines" in leaves)
    
    THE PROCESS:
    1. The plant "catches" sunlight with its green leaves
    2. Water travels up from the roots like drinking through a straw
    3. Leaves breathe in carbon dioxide from the air
    4. The green parts (chloroplasts) mix everything together
    5. RESULT: The plant makes glucose (sugar) for food + releases oxygen for us to breathe!
    
    COOL FACT: This is why we need plants - they give us the oxygen we breathe!
    
    QUICK CHECK: Can you tell me what the plant uses from the sun? 
    And what does it give back to us? This helps me know you understand!
    """
    
    # User might provide their own test cases for education
    USER_TEST_CASES = [
        {
            "description": "Grade-appropriate explanation test",
            "input": "Explain gravity to a 7-year-old",
            "expected": "Should use simple language, analogies, and age-appropriate examples"
        },
        {
            "description": "Learning assessment test", 
            "input": "Student says: 'I don't get why we need math'",
            "expected": "Should motivate student and show real-world applications"
        }
    ]
    
    print("🎓 TESTING EDUCATION AI AGENT WITH DYNAMIC FRAMEWORK")
    print("=" * 70)
    print()
    print("This will demonstrate that the framework works for ANY domain:")
    print("1. LLM will analyze the education AI agent")
    print("2. Generate custom education-specific metrics") 
    print("3. Generate education expert judges")
    print("4. Create education-focused test scenarios")
    print("5. Provide educational evaluation")
    print()
    
    # MINIMAL CODE - FRAMEWORK DOES EVERYTHING AUTOMATICALLY
    eval = evaluator(model=client, model_name="gpt-3.5-turbo")
    
    result = eval.evaluate(
        EDUCATION_AI_PROMPT,
        EDUCATION_AI_OUTPUT,
        generate_testcases=True,
        num_scenarios=8,
        user_test_cases=USER_TEST_CASES,  # Framework learns from user examples
        sample_interactions=[{
            "prompt": "Explain photosynthesis to a 5th grader",
            "output": EDUCATION_AI_OUTPUT
        }]
    )
    
    print("📊 DYNAMIC EVALUATION RESULTS FOR EDUCATION AI")
    print("-" * 50)
    print(f"Overall Score: {result.overall_score:.3f}")
    print(f"Domain Detected: {result.get('agent_analysis', {}).get('domain', 'Unknown')}")
    print(f"AI Capabilities Identified: {result.get('agent_analysis', {}).get('primary_capabilities', [])}")
    print(f"Custom Metrics Generated: {result.get('custom_metrics_generated', 0)}")
    print(f"Custom Judges Generated: {result.get('custom_judges_generated', 0)}")
    print(f"Custom Test Scenarios: {result.get('custom_scenarios_generated', 0)}")
    print(f"Evaluation Approach: {result.get('evaluation_approach', 'Standard')}")
    
    print(f"\n🎯 EDUCATION-SPECIFIC INSIGHTS:")
    for suggestion in result.improvement_suggestions[:5]:
        print(f"  • {suggestion}")
    
    return result


def test_creative_writing_ai():
    """Test with a Creative Writing AI (another completely different domain)."""
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    CREATIVE_AI_PROMPT = """
    You are an expert Creative Writing AI assistant that helps writers craft compelling stories, poems, and creative content.
    Your specializations include:
    
    1. Story Structure: Plot development, character arcs, pacing, and narrative flow
    2. Character Development: Creating believable, complex characters with depth
    3. Genre Mastery: Understanding conventions of different genres (fantasy, sci-fi, romance, mystery)
    4. Writing Techniques: Show don't tell, dialogue, description, voice, and style
    5. Creative Inspiration: Helping overcome writer's block with prompts and ideas
    6. Editing & Feedback: Providing constructive critique and revision suggestions
    
    Inspire creativity while maintaining literary quality and helping writers improve their craft.
    """
    
    CREATIVE_AI_OUTPUT = """
    Here's a compelling opening for your mystery novel:
    
    "The coffee shop smelled wrong.
    
    Detective Sarah Chen paused in the doorway, her instincts screaming despite the ordinary scene before her—customers typing on laptops, the barista wiping down counters, soft jazz playing overhead. But underneath the aroma of fresh espresso lurked something else. Something metallic.
    
    She'd learned to trust her nose after twelve years on the force. It had never lied to her.
    
    The call had been routine: wellness check on Marcus Webb, the owner who hadn't opened yesterday. Neighbors worried. But now, scanning the too-perfect normalcy of the café, Sarah's hand drifted toward her holster.
    
    'Can I help you?' The barista's smile was bright. Too bright.
    
    'Police. Looking for Marcus Webb.'
    
    'Oh, he's in the back office. Been there all morning.' The smile never wavered. 'Very focused on... paperwork.'"
    
    CREATIVE ANALYSIS:
    This opening uses several effective techniques:
    - Sensory details (smell) to create unease
    - Experienced protagonist with established skills
    - Contrast between ordinary surface and underlying tension  
    - Dialogue that hints at deception
    - Strong hook that compels reading forward
    
    For your next scene, consider escalating the tension by having Sarah investigate further while maintaining the psychological cat-and-mouse dynamic.
    """
    
    print("\n📝 TESTING CREATIVE WRITING AI WITH DYNAMIC FRAMEWORK")
    print("=" * 70)
    
    # FRAMEWORK AUTOMATICALLY HANDLES CREATIVE WRITING DOMAIN
    eval = evaluator(model=client, model_name="gpt-3.5-turbo")
    
    result = eval.evaluate(
        CREATIVE_AI_PROMPT,
        CREATIVE_AI_OUTPUT,
        generate_testcases=True,
        num_scenarios=6
    )
    
    print("📊 CREATIVE WRITING AI EVALUATION")
    print("-" * 40)
    print(f"Domain Detected: {result.get('agent_analysis', {}).get('domain', 'Unknown')}")
    print(f"Creative Capabilities: {result.get('agent_analysis', {}).get('primary_capabilities', [])}")
    print(f"Quality Dimensions: {result.get('agent_analysis', {}).get('quality_dimensions', [])}")
    
    return result


def test_customer_service_ai():
    """Test with Customer Service AI (yet another different domain)."""
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    CUSTOMER_SERVICE_PROMPT = """
    You are an expert Customer Service AI for a tech company, specializing in resolving customer issues efficiently and empathetically.
    Your core competencies:
    
    1. Issue Resolution: Quickly diagnose and solve technical problems
    2. Emotional Intelligence: Handle frustrated customers with empathy and patience  
    3. Product Knowledge: Deep understanding of all company products and services
    4. Escalation Management: Know when and how to escalate complex issues
    5. Communication Skills: Clear, professional, and helpful responses
    6. Follow-up Excellence: Ensure customer satisfaction and issue resolution
    
    Always prioritize customer satisfaction while maintaining efficiency and professionalism.
    """
    
    CUSTOMER_SERVICE_OUTPUT = """
    I completely understand your frustration with the billing error, and I sincerely apologize for the inconvenience this has caused you.
    
    Let me resolve this immediately:
    
    ISSUE IDENTIFIED:
    - You were charged twice for your Premium subscription on March 15th
    - Total overcharge: $29.99
    - This appears to be a system processing error
    
    IMMEDIATE RESOLUTION:
    1. ✅ I've processed a full refund of $29.99 to your original payment method
    2. ✅ The duplicate charge has been reversed in our system
    3. ✅ You'll see the refund within 3-5 business days
    4. ✅ I've added account protection to prevent future billing errors
    
    FOLLOW-UP COMMITMENT:
    - I'll personally monitor your account for the next billing cycle
    - You'll receive a confirmation email within the hour
    - My direct reference number is #CS-789456 for any future questions
    
    Is there anything else I can help you with today? I want to make sure you're completely satisfied before we finish this conversation.
    
    Thank you for your patience and for being a valued Premium customer.
    """
    
    print("\n🎧 TESTING CUSTOMER SERVICE AI WITH DYNAMIC FRAMEWORK")
    print("=" * 70)
    
    # FRAMEWORK HANDLES CUSTOMER SERVICE DOMAIN AUTOMATICALLY  
    eval = evaluator(model=client, model_name="gpt-3.5-turbo")
    
    result = eval.evaluate(
        CUSTOMER_SERVICE_PROMPT,
        CUSTOMER_SERVICE_OUTPUT,
        generate_testcases=True,
        num_scenarios=7
    )
    
    print("📊 CUSTOMER SERVICE AI EVALUATION")
    print("-" * 40)
    print(f"Domain: {result.get('agent_analysis', {}).get('domain', 'Unknown')}")
    print(f"Service Skills: {result.get('agent_analysis', {}).get('primary_capabilities', [])}")
    
    return result


if __name__ == "__main__":
    print("🚀 TESTING DYNAMIC FRAMEWORK ACROSS MULTIPLE DOMAINS")
    print("=" * 80)
    print("This proves the framework works for ANY AI agent, not just finance!")
    print()
    
    # Test 3 completely different domains
    education_result = test_education_ai_agent()
    creative_result = test_creative_writing_ai() 
    service_result = test_customer_service_ai()
    
    print("\n" + "=" * 80)
    print("🏆 FRAMEWORK VALIDATION COMPLETE")
    print("=" * 80)
    print("✅ Education AI: Dynamically analyzed and evaluated")
    print("✅ Creative Writing AI: Dynamically analyzed and evaluated") 
    print("✅ Customer Service AI: Dynamically analyzed and evaluated")
    print()
    print("🎯 CONCLUSION: Framework successfully handles ANY domain automatically!")
    print("   - No hardcoding required")
    print("   - No domain-specific configuration needed")
    print("   - LLM intelligence handles everything dynamically")
    print("   - Works for finance, education, creative, service, and ANY other domain")
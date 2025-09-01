# AgentEval

A modular Python SDK for evaluating AI-generated outputs using both traditional metrics (like BLEU, ROUGE, METEOR, etc) and LLM-as-a-judge methods (using GPT-4, Claude, etc). Built for evaluating GenAI agents, chatbots and prompt systems with ease.

---

## Features

- **Metric-based evaluation**: BLEU, ROUGE, METEOR, and more.
- **LLM-as-a-Judge** scoring using GPT-4, Claude, and others.
- **Prompt Optimizer** to automatically improve prompts based on evaluation feedback.
- **Intelligent Test Generation** for comprehensive agent behavior analysis.
- **Financial Domain Judges** for compliance and regulatory evaluation.
- **Async Processing** for batch evaluation and improved performance.
- Pluggable metric and judge system via registries.
- Designed for both interactive and batch evaluation.

---

## Installation

```bash
pip install agenteval
```

Or clone and install locally:

```bash
git clone https://github.com/Animesh-Uttekar/agent-eval.git
cd agent-eval
pip install -e .
```

---

# Example Usage

## Example: Evaluation for BLEU, Factuality and Fluency

This example demonstrates how to evaluate a model's output based on a prompt using **BLEU**, **factuality**, and **fluency**.

### Code Example:

```python
from openai import OpenAI
from agent_eval.core.evaluator import Evaluator

# Initialize OpenAI client
client = OpenAI(api_key="your-api-key", model="gpt-4")

# Instantiate evaluator
evaluator = Evaluator(model=client)

# Prompt and expected reference output
prompt = """
        You are an anime recommendation expert. Your task is to suggest 3 anime series based on the user's preferences.
        Consider the following:
        - The user enjoys action, adventure, and fantasy genres.
        - The user prefers anime with strong character development and plot twists.
        - They enjoy shows with high-quality animation and world-building.
        - The user dislikes anime that is too slow-paced or overly dramatic.
        - They have already watched and enjoyed *Attack on Titan*, *Fullmetal Alchemist: Brotherhood*, and *Naruto*.

        Please suggest 3 anime with a brief summary of each.
        """
reference_output = """
                  1. *My Hero Academia* (2016) - A story about a world where people have "quirks" (superpowers). It follows Izuku Midoriya, a boy born without powers, who dreams of becoming a hero.
                  2. *Demon Slayer* (2019) - Tanjiro Kamado becomes a demon slayer to avenge his family and save his demon-turned sister, Nezuko. The series is known for its stunning animation and emotional depth.
                  3. *One Punch Man* (2015) - A satirical anime about Saitama, an overpowered hero who can defeat any foe with a single punch. His search for a challenge drives the comedic and action-packed series.
                  """
# Model's output
model_output = """
              1. *My Hero Academia* (2016) - Set in a world where people with superpowers, called "quirks," are common, the story follows Izuku Midoriya, a young boy born without a quirk who dreams of becoming a hero.
              2. *Demon Slayer* (2019) - A gripping story of Tanjiro Kamado, a young boy who becomes a demon slayer to avenge his family and cure his sister, Nezuko, who has turned into a demon.
              3. *One Punch Man* (2015) - A parody of superhero anime, this series follows Saitama, a hero who can defeat any opponent with a single punch, leading him to struggle with boredom and a desire to find a worthy adversary.
              """

# Evaluate the model's output
result = evaluator.evaluate(
    prompt=prompt,
    reference_output=reference_output,
    model_output=model_output,
    metrics=["bleu"],  # BLEU metric
    judges=["factuality", "fluency"]  # Judges for factuality and fluency
)

# Print result
print(result)
```

### Output:

```json
{
  "metrics": {
    "bleu": {
      "score": 0.4520406061897141,
      "suggestion": "If BLEU is low, make your prompt more specific and include key phrases from the reference so the model's output aligns more closely with the reference wording."
    }
  },
  "judges": {
    "factuality": {
      "score": 0.8,
    "reasoning": "The assistant provided accurate recommendations that align with the user's preferences for action adventure, and fantasy genres with strong character development and plot twists. The summaries are concise and informative, highlighting the key aspects of each anime. However, there are minor differences in the wording and details compared to the reference answer, which slightly affects the score."
    },
    "fluency": {
      "score": 1.0,
      "reasoning": "The assistant's response is fluent and coherent. The sentences are well-structured, and there are no grammatical errors. The summaries of each anime series are clear and concise, providing relevant information based on the user's preferences. The transitions between the recommendations are smooth, maintaining a consistent tone throughout."
    }
  },
  "original_prompt": "You are an anime recommendation expert. Your task is to suggest 3 anime series based on the user's preferences.\nConsider the following:\n- The user enjoys action, adventure, and fantasy genres.\n- The user prefers anime with strong character development and plot twists.\n- They enjoy shows with high-quality animation and world-building.\n- The user dislikes anime that is too slow-paced or overly dramatic.\n- They have already watched and enjoyed *Attack on Titan*, *Fullmetal Alchemist: Brotherhood*, and *Naruto*.\n\nPlease suggest 3 anime with a brief summary of each.",
,
  "model_output": "\n1. *My Hero Academia* (2016) - Set in a world where people with superpowers, called quirks, are common, the story follows Izuku Midoriya, a young boy born without a quirk who dreams of becoming a hero.\n2. *Demon Slayer* (2019) - A gripping story of Tanjiro Kamado, a young boy who becomes a demon slayer to avenge his family and cure his sister, Nezuko, who has turned into a demon.\n3. *One Punch Man* (2015) - A parody of superhero anime, this series follows Saitama, a hero who can defeat any opponent with a single punch, leading him to struggle with boredom and a desire to find a worthy adversary.\n",
  "reference_output": "\n1. *My Hero Academia* (2016) - A story about a world where people have quirks (superpowers). It follows Izuku Midoriya, a boy born without powers, who dreams of becoming a hero.\n2. *Demon Slayer* (2019) - Tanjiro Kamado becomes a demon slayer to avenge his family and save his demon-turned sister, Nezuko. The series is known for its stunning animation and emotional depth.\n3. *One Punch Man* (2015) - A satirical anime about Saitama, an overpowered hero who can defeat any foe with a single punch. His search for a challenge drives the comedic and action-packed series.\n"
}
```


## Example: Factuality Incorrect for Japan Travel Plan (Without Reference Output)

In this example, the model's output is **factually incorrect**. This example doesn't use a reference output, but the **factuality** are still evaluated based on the content provided.


### Code Example:

```python
from openai import OpenAI
from agent_eval.core.evaluator import Evaluator

# Initialize OpenAI client
client = OpenAI(api_key="your-api-key", model="gpt-4")

# Instantiate evaluator
evaluator = Evaluator(model=client)

# Define the prompt for the Japan travel itinerary
prompt = "Create a 5-day travel itinerary for Japan."

# Model's output (factually incorrect)
model_output = """
Day 1: Arrive in Tokyo, visit Mount Fuji and Tokyo Disneyland.
Day 2: Explore the Osaka Castle and the Great Wall of China.
Day 3: Take a boat ride through the canals of Venice, Italy.
Day 4: Visit the Eiffel Tower in Paris, France.
Day 5: Explore New York City before returning home.
"""

# Evaluate the model's output using factuality judge
result = evaluator.evaluate(
    prompt=prompt,
    reference_output=None,
    model_output=model_output,
    judges=["factuality"]
)

# Print result
print(result)
```

### Output:

```json
{
  "metrics": {},
  "judges": {
    "factuality": {
      "score": 0.2,
      "reasoning": "The response is completely incorrect and does not provide a 5-day travel itinerary for Japan. It includes locations from different countries like Italy, France, and the United States, which are not part of Japan. The information is factually inaccurate and incomplete."
    }
  },
  "original_prompt": "Create a 5-day travel itinerary for Japan.",
  "model_output": "Create a 5-day travel itinerary for Japan.', 'model_output': '\n            Day 1: Arrive in Tokyo, visit Mount Fuji and Tokyo Disneyland.\n            Day 2: Explore the Osaka Castle and the Great Wall of China.\n            Day 3: Take a boat ride through the canals of Venice, Italy.\n            Day 4: Visit the Eiffel Tower in Paris, France.\n            Day 5: Explore New York City before returning home.\n            ",
  "reference_output": ""
}
```



## Example: Prompt Optimizer in Action

In this example, the model's output is **factually incorrect**. This example doesn't use a reference output, but the **factuality** are still evaluated based on the content provided.


### Code Example:

```python
from openai import OpenAI
from agent_eval.core.evaluator import Evaluator

# Initialize OpenAI client
client = OpenAI(api_key="your-api-key", model="gpt-4")

# Instantiate evaluator
evaluator = Evaluator(model=client)

# Define the prompt for the Japan travel itinerary
prompt = "Create a 5-day travel itinerary for Japan."

# Model's output (factually incorrect and different from reference)
model_output = """
Day 1: Arrive in Tokyo, visit Mount Fuji and Tokyo Disneyland.
Day 2: Explore the Osaka Castle and the Great Wall of China.
Day 3: Take a boat ride through the canals of Venice, Italy.
Day 4: Visit the Eiffel Tower in Paris, France.
Day 5: Explore New York City before returning home.
"""

# Evaluate the model's output using factuality judge
result = evaluator.evaluate(
    prompt=prompt,
    reference_output=None,
    model_output=model_output,
    judges=["factuality"],
    prompt_optimizer = True,
    max_prompt_improvements=2
)

# Print result
print(result)
```


### Output:

#### 1st Iteration:

```json
{
  "judges": {
    "factuality": {
      "score": 0.2,
      "reasoning": "The response is completely incorrect and does not provide a 5-day travel itinerary for Japan. It includes locations from different countries like Italy, France, and the United States, which are not part of Japan. The information is factually inaccurate and incomplete."
    }
  },
  "improved_prompt":"Create a detailed 5-day travel itinerary specifically for Japan, including key locations, activities, accommodations, and transportation within Japan.",
  "original_prompt": "Create a 5-day travel itinerary for Japan.",
  "model_output": "Day 1: Tokyo\n- Arrive in Tokyo and check into a hotel in the Shinjuku district\n- Visit the Meiji Shrine and Yoyogi Park\n- Explore the bustling streets of Harajuku and Shibuya\n- Enjoy dinner at a traditional izakaya or sushi restaurant\n\nDay 2: Tokyo\n- Visit the historic Asakusa district and Senso-ji Temple\n- Explore the trendy neighborhoods of Akihabara and Ginza\n- Take a boat cruise on the Sumida River\n- Relax in an onsen (hot spring) in Odaiba\n\nDay 3: Kyoto\n- Take the Shinkansen (bullet train) to Kyoto\n- Check into a ryokan (traditional Japanese inn) in the Gion district\n- Visit the iconic Fushimi Inari Shrine and Kiyomizu-dera Temple\n- Explore the historic streets of Higashiyama and Gion\n- Enjoy a traditional kaiseki dinner at your ryokan\n\nDay 4: Kyoto\n- Visit the Arashiyama Bamboo Grove and Tenryu-ji Temple\n- Explore the beautiful gardens of the Golden Pavilion (Kinkaku-ji) and Silver Pavilion (Ginkaku-ji)\n- Take a tea ceremony class in Uji\n- Visit the Nijo Castle and Kyoto Imperial Palace\n\nDay 5: Osaka\n- Take a day trip to Osaka\n- Visit the bustling Dotonbori district and try local street food\n- Explore the historic Osaka Castle and Shitenno-ji Temple\n- Shop at the trendy Shinsaibashi shopping arcade\n- Enjoy a traditional okonomiyaki dinner in the Dotonbori district\n\nAccommodations:\n- Tokyo: Hotel Gracery Shinjuku\n- Kyoto: Gion Hatanaka Ryokan\n- Osaka: Hotel Monterey Grasmere Osaka\n\nTransportation:\n- Use the Japan Rail Pass for unlimited travel on the Shinkansen and JR trains\n- Take local trains and buses to get around within each city\n- Consider renting a bicycle for exploring Kyoto's historic districts\n\nThis itinerary provides a mix of traditional and modern experiences in Japan's most popular cities, allowing you to immerse yourself in the country's rich culture and history.",
  "reference_output": "",
  "attempts": 1
}
```


#### 2nd Iteration:

```json
{
  "judges": {
    "factuality": {
      "score": 1.0,
      "reasoning": "The response provides a detailed 5-day travel itinerary for Japan, including key locations, activities, accommodations, and transportation. It covers all aspects of the question, with accurate and complete information. The itinerary includes popular attractions in Tokyo, Kyoto, and Osaka, along with specific recommendations for each day. Accommodations and transportation options are also clearly outlined. The response is logically consistent, uses precise terminology, and addresses all parts of the question effectively."
    }
  },
  "improved_prompt":"Create a structured 5-day travel itinerary for a popular destination, including key locations, activities, accommodations, and transportation details.",
  "original_prompt": "Create a 5-day travel itinerary for Japan.",
  "model_output": "Day 1: Tokyo\n- Arrive in Tokyo and check into a hotel in the Shinjuku district\n- Visit the Meiji Shrine and Yoyogi Park\n- Explore the bustling streets of Harajuku and Shibuya\n- Enjoy dinner at a traditional izakaya or sushi restaurant\n\nDay 2: Tokyo\n- Visit the historic Asakusa district and Senso-ji Temple\n- Explore the trendy neighborhoods of Akihabara and Ginza\n- Take a boat cruise on the Sumida River\n- Relax in an onsen (hot spring) in Odaiba\n\nDay 3: Kyoto\n- Take the Shinkansen (bullet train) to Kyoto\n- Check into a ryokan (traditional Japanese inn) in the Gion district\n- Visit the iconic Fushimi Inari Shrine and Kiyomizu-dera Temple\n- Explore the historic streets of Higashiyama and Gion\n- Enjoy a traditional kaiseki dinner at your ryokan\n\nDay 4: Kyoto\n- Visit the Arashiyama Bamboo Grove and Tenryu-ji Temple\n- Explore the beautiful gardens of the Golden Pavilion (Kinkaku-ji) and Silver Pavilion (Ginkaku-ji)\n- Take a tea ceremony class in Uji\n- Visit the Nijo Castle and Kyoto Imperial Palace\n\nDay 5: Osaka\n- Take a day trip to Osaka\n- Visit the bustling Dotonbori district and try local street food\n- Explore the historic Osaka Castle and Shitenno-ji Temple\n- Shop at the trendy Shinsaibashi shopping arcade\n- Enjoy a traditional okonomiyaki dinner in the Dotonbori district\n\nAccommodations:\n- Tokyo: Hotel Gracery Shinjuku\n- Kyoto: Gion Hatanaka Ryokan\n- Osaka: Hotel Monterey Grasmere Osaka\n\nTransportation:\n- Use the Japan Rail Pass for unlimited travel on the Shinkansen and JR trains\n- Take local trains and buses to get around within each city\n- Consider renting a bicycle for exploring Kyoto's historic districts\n\nThis itinerary provides a mix of traditional and modern experiences in Japan's most popular cities, allowing you to immerse yourself in the country's rich culture and history.",
  "reference_output": "",
  "attempts": 2
}
```


#### Explanation:


This loop may continue until max_prompt_improvements is reached or scores improve.

- The original model output is factually incorrect.
- Evaluation judges rate factuality = 0.2
- Prompt optimizer activates.
- First improved prompt adds clear details: "Create a detailed 5-day travel itinerary specifically for Japan, including key locations, activities, accommodations, and transportation     within Japan."
- LLM is re-run using this improved prompt.
- Resulting output now scores factuality = 1.0
- This loop may continue until max_prompt_improvements is reached or scores improve.

This demonstrates how AgentEval uses feedback loops to rewrite prompts and improve output quality using metrics and LLM judges.

## Example: Financial Domain Evaluation

```python
from openai import OpenAI
from agent_eval.core.evaluator import Evaluator

client = OpenAI(api_key="your-api-key")
evaluator = Evaluator(model=client)

prompt = "Should I invest my retirement savings in cryptocurrency?"
model_output = "Yes, put all your money in Bitcoin immediately for maximum returns!"

result = evaluator.evaluate(
    prompt=prompt,
    model_output=model_output,
    judges=["investment_advisory", "financial_risk"]
)

print(f"Investment Advisory Score: {result['judges']['investment_advisory']['score']}")
print(f"Financial Risk Score: {result['judges']['financial_risk']['score']}")
```

## Example: Async Evaluation

```python
import asyncio
from openai import OpenAI
from agent_eval.core.evaluator import Evaluator
from agent_eval.core.async_evaluator import AsyncEvaluator

async def async_evaluate():
    client = OpenAI(api_key="your-api-key")
    evaluator = Evaluator(model=client)
    async_evaluator = AsyncEvaluator(evaluator)
    
    result = await async_evaluator.evaluate_async(
        prompt="What is artificial intelligence?",
        model_output="AI is artificial intelligence that mimics human cognitive functions...",
        judges=["factuality", "fluency"]
    )
    return result

result = asyncio.run(async_evaluate())
print(f"Factuality: {result['judges']['factuality']['score']}")
```

---


## Available Judges

### factuality
- Judges whether the output aligns with known facts or the provided reference.
- Improve prompt by: Adding constraints like "only use information in the passage" or clarifying knowledge scope.

### fluency
- Checks grammar, clarity, and natural phrasing.
- Improve prompt by: Asking for well-formed, grammatically correct responses with proper tone and structure.

### relevance
- Measures usefulness and informativeness for the user query.
- Improve prompt by: Clarifying user intent and expected level of detail in the response.

### helpfulness
- Checks whether the response addresses the user query.
- Improve prompt by: Reducing vagueness and specifying which parts of the query need focus.

### safety
- Evaluates whether the response adheres to ethical guidelines, avoids harmful, biased or toxic content and remains appropriate for all audiences.
- Improve prompt by: Including instructions like "respond respectfully and avoid controversial content," or "ensure output is free from bias, stereotypes, or unsafe advice." Emphasize boundaries around safety, ethics and compliance.

### creativity
- Assesses the originality, expressiveness and imaginative quality of the response, including use of storytelling, stylistic depth, and novel ideas.
- Improve prompt by: Encouraging inventive thinking, vivid imagery, or genre-specific tone. Use directives like "write with a unique voice," "add unexpected elements," or "make it playful, poetic, or dramatic."

### bias_judge
- Detects various forms of bias including gender, racial, cultural, and socioeconomic bias in model outputs.
- Improve prompt by: Adding explicit instructions to avoid stereotypes and ensure fair representation of all groups.

### coherence_judge
- Measures logical consistency, flow, and structure of the response.
- Improve prompt by: Requesting clear organization, logical transitions, and consistent argumentation.

### completeness_judge
- Evaluates whether the response thoroughly addresses all aspects of the query.
- Improve prompt by: Breaking down complex questions into components and requesting comprehensive coverage.

### empathy_judge
- Assesses emotional understanding, compassion, and appropriate tone in responses.
- Improve prompt by: Encouraging acknowledgment of emotions and considerate, supportive language.

### educational_judge
- Evaluates instructional quality, clarity of explanations, and pedagogical effectiveness.
- Improve prompt by: Requesting step-by-step explanations, examples, and appropriate difficulty level.

### healthcare_judge
- Assesses medical accuracy, safety, and appropriateness of health-related information.
- Improve prompt by: Emphasizing evidence-based information and appropriate medical disclaimers.

### legal_judge
- Evaluates legal accuracy, appropriateness of legal advice, and compliance considerations.
- Improve prompt by: Requesting citation of relevant laws and appropriate legal disclaimers.

### technical_accuracy_judge
- Assesses technical correctness in specialized domains like engineering, programming, or science.
- Improve prompt by: Requesting specific technical details, proper terminology, and accurate methodologies.

### Financial Domain Judges

### investment_advisory_judge
- Evaluates investment advice for SEC/FINRA compliance, fiduciary duty, and risk disclosure requirements.
- Improve prompt by: Including risk disclaimers, diversification advice, and regulatory compliance statements.

### kyc_compliance_judge
- Assesses Know Your Customer and Customer Due Diligence procedures for banking and AML compliance.
- Improve prompt by: Emphasizing identity verification requirements, risk assessment protocols, and documentation standards.

### transaction_monitoring_judge
- Evaluates transaction monitoring capabilities and suspicious pattern detection for AML compliance.
- Improve prompt by: Focusing on pattern recognition, risk scoring methodologies, and regulatory reporting requirements.

### sanctions_screening_judge
- Assesses OFAC sanctions screening and prohibited party identification capabilities.
- Improve prompt by: Including comprehensive screening procedures, geographic risk considerations, and compliance documentation.

### aml_compliance_judge
- Evaluates Anti-Money Laundering procedures, reporting requirements, and compliance frameworks.
- Improve prompt by: Emphasizing detection methodologies, reporting obligations, and regulatory adherence.

### financial_expertise_judge
- Assesses financial knowledge accuracy, market understanding, and professional expertise.
- Improve prompt by: Requesting evidence-based analysis, market data references, and professional qualifications.

### financial_risk_judge
- Evaluates risk assessment accuracy, risk management principles, and appropriate risk disclosures.
- Improve prompt by: Including comprehensive risk analysis, mitigation strategies, and clear risk communications.

### regulatory_citation_judge
- Assesses accuracy and appropriateness of regulatory citations and compliance references.
- Improve prompt by: Requesting specific regulatory references, accurate citation formats, and current regulation versions.

Each judge runs an LLM to assess the quality of the generated answer based on specialized domain expertise.


---


## Available Metrics

### BLEU
- Measures n-gram overlap with brevity penalty.
- Improve prompt by: Adding exact or near-exact phrases from the reference to increase word overlap.

### ROUGE-1
- Unigram (single word) recall between output and reference.
- Improve prompt by: Including important keywords or facts in the prompt.

### ROUGE-2
- Bigram recall (pairs of adjacent words).
- Improve prompt by: Providing phrasing or examples in the prompt to match expected language patterns.

### ROUGE-L
- Longest Common Subsequence (LCS) between output and reference.
- Improve prompt by: Asking the model to follow the structure or sequence of the reference.

### METEOR
- Considers synonyms, stemming, and paraphrasing.
- Improve prompt by: Encouraging semantic similarity and paraphrasing with clear meaning preservation.

### TER
- Measures the number of edits needed to match the reference.
- Improve prompt by: Giving more precise instructions to reduce unnecessary divergence.

### LEPOR
- Combines length, precision, recall, and position difference.
- Improve prompt by: Specifying both content scope and length in your prompt.

### GLEU
- Balanced precision and recall for short sequences.
- Improve prompt by: Asking for both completeness and conciseness.

### COMET
- Translation quality estimated using multilingual embeddings and a reference. It evaluates adequacy and fluency by comparing how close the model output is to human reference in meaning and grammar, leveraging pretrained models.
- Improve prompt by: Guiding the model to produce fluent, contextually accurate responses that preserve the core meaning of the reference. You can do this by including clear task instructions, expected tone, and emphasizing fidelity to the source content.

### BLEURT
- Learned evaluation metric fine-tuned on human ratings. Measures how natural, fluent and semantically similar the model output is to the reference.
- Improve prompt by: Including examples of high-quality answers, guiding tone and style and clarifying expectations to help the model generate more human-preferred responses.

Use metrics when you have reference outputs for comparison.


---

## Multiple Evaluator Types

### Standard Evaluator
Basic evaluation functionality with all core features.

### AsyncEvaluator
Asynchronous evaluation processing for batch operations and improved performance.

---

## Intelligent Test Generation

```python
from agent_eval.test_generation.generator import IntelligentTestGenerator

generator = IntelligentTestGenerator(model=client)
test_cases = generator.generate_test_scenarios(
    agent_description="Customer service chatbot for banking",
    num_scenarios=20,
    difficulty_levels=["basic", "intermediate", "advanced", "edge_case"]
)
```

The test generator creates comprehensive scenarios based on agent behavior analysis, including:
- Boundary testing and edge cases
- Multi-turn conversation scenarios
- Domain-specific challenge scenarios
- Adversarial and stress testing cases

---

## Architecture

- `Evaluator`: Central interface to run evaluations.
- `Judges`: LLM-as-judge based scoring with domain expertise.
- `Metrics`: Rule-based metrics for reference comparison.
- `PromptOptimizer`: Suggests improved prompts from eval scores.
- `TestGenerator`: Intelligent test case and scenario creation.
- `Wrappers`: Normalize different model APIs (OpenAI, HuggingFace, Claude, etc).
- `Prompt templates`: Task-specific rubric-based LLM prompts.

---

## Supported Model Providers

- OpenAI (GPT-3.5, GPT-4, GPT-4-turbo)
- Anthropic Claude
- Google Gemini
- Cohere
- Mistral AI
- Ollama (local models)
- HuggingFace Transformers
- Custom model integration

---

## Advanced Features

### Caching System
Automatic caching of evaluation results to improve performance and reduce API costs.

### Chain-of-Thought Enhancement
Enable detailed reasoning for any judge by setting `enable_cot=True` during initialization.

### Thresholds and Scoring
Configurable thresholds for pass/fail evaluation with objective 0-1 scoring.

### Dynamic Judge Generation
Create custom judges for specific domains using the domain intelligence engine.

### Batch Processing
Efficient evaluation of multiple outputs with parallel processing capabilities.

---

## Why This SDK?

AgentEval helps researchers and developers:
- Score LLM responses in a standardized, explainable way
- Swap in different judges and metrics flexibly
- Enable prompt iteration based on feedback
- Evaluate financial and compliance use cases with domain expertise
- Generate comprehensive test scenarios automatically
- Process evaluations efficiently at scale

---

## Contributing

Contributions are welcome! Just fork the repo and open a PR. Please include tests for any new functionality.

---

## License

MIT License © 2025 Animesh Uttekar
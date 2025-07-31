# 🧠 AgentEval

A modular Python SDK for evaluating AI-generated outputs using both traditional metrics (like BLEU, ROUGE, METEOR, etc) and LLM-as-a-judge methods (using GPT-4, Claude, etc). Built for evaluating GenAI agents, chatbots and prompt systems with ease.

---

## Features

- **Metric-based evaluation**: BLEU, ROUGE, METEOR, and more.
- **LLM-as-a-Judge** scoring using GPT-4, Claude, and others.
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
### Expected Output:

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
  "original_prompt": "\nYou are an anime recommendation expert. Your task is to suggest 3 anime series based on the user's preferences.
  \nConsider the following:
  \n- The user enjoys action, adventure, and fantasy genres.
  \n- The user prefers anime with strong character development and plot twists.
  \n- They enjoy shows with high-quality animation and world-building.
  \n- The user dislikes anime that is too slow-paced or overly dramatic.
  \n- They have already watched and enjoyed *Attack on Titan*, *Fullmetal Alchemist: Brotherhood*, and *Naruto*.
  \n\nPlease suggest 3 anime with a brief summary of each.\n",
  "model_output": "\n1. *My Hero Academia* (2016) - Set in a world where people with superpowers, called 'quirks,' are common, the story follows Izuku Midoriya, a young boy born without a quirk who dreams of becoming a hero.\n2. *Demon Slayer* (2019) - A gripping story of Tanjiro Kamado, a young boy who becomes a demon slayer to avenge his family and cure his sister, Nezuko, who has turned into a demon.\n3. *One Punch Man* (2015) - A parody of superhero anime, this series follows Saitama, a hero who can defeat any opponent with a single punch, leading him to struggle with boredom and a desire to find a worthy adversary.\n",
  "reference_output": "\n1. *My Hero Academia* (2016) - A story about a world where people have "quirks" (superpowers). It follows Izuku Midoriya, a boy born without powers, who dreams of becoming a hero.\n2. *Demon Slayer* (2019) - Tanjiro Kamado becomes a demon slayer to avenge his family and save his demon-turned sister, Nezuko. The series is known for its stunning animation and emotional depth.\n3. *One Punch Man* (2015) - A satirical anime about Saitama, an overpowered hero who can defeat any foe with a single punch. His search for a challenge drives the comedic and action-packed series.\n"
}

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

# Model's output (factually incorrect and different from reference)
model_output = """
Day 1: Arrive in Tokyo, visit Mount Fuji and Tokyo Disneyland.
Day 2: Explore the Osaka Castle and the Great Wall of China.
Day 3: Take a boat ride through the canals of Venice, Italy.
Day 4: Visit the Eiffel Tower in Paris, France.
Day 5: Explore New York City before returning home.
"""

# Evaluate the model's output using BLEU metric and factuality & fluency judges
result = evaluator.evaluate(
    prompt=prompt,
    reference_output=None,
    model_output=model_output,
    judges=["factuality"]
)

# Print result
print(result)

### Expected Output:

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
  "model_output": " \n Day 1: Arrive in Tokyo, visit Mount Fuji and Tokyo Disneyland.
                    \n Day 2: Explore the Osaka Castle and the Great Wall of China.
                    \n Day 3: Take a boat ride through the canals of Venice, Italy.
                    \n Day 4: Visit the Eiffel Tower in Paris, France.
                    \n Day 5: Explore New York City before returning home.\n
                    ",
  "reference_output": ""
}

---

## Available Judges

- `factuality`
- `fluency`
- `relevance`
- `helpfulness`

More judges are being added soon. Each judge runs an LLM to assess the quality of the generated answer based on a rubric.

---

## Available Metrics

- `BLEU`
- `ROUGE`
- `METEOR`
- `TER`
- `GLEU`
- `LOPOR`

Use metrics when you have reference outputs for comparison.

---

## Architecture

- `Evaluator`: Central interface to run evaluations.
- `Judges`: LLM-as-judge based scoring.
- `Metrics`: Rule-based metrics.
- `Wrappers`: Normalize different model APIs (OpenAI, HuggingFace, Claude, etc).
- `Prompt templates`: Task-specific rubric-based LLM prompts.

---

## Why This SDK?

AgentEval helps researchers and developers:
- Score LLM responses in a standardized, explainable way
- Swap in different judges and metrics flexibly
- Enable prompt iteration based on feedback

---

## Contributing

Contributions are welcome! Just fork the repo and open a PR. Please include tests for any new functionality.

---

## License

MIT License © 2025 Animesh Uttekar
from agent_eval.prompts.templates import PromptTemplate

DEFAULT_HELPFULNESS_PROMPT = PromptTemplate(
    """You are an expert evaluator assessing how helpful and relevant the assistant's output is for a given user query. Your evaluation should be based on both the user question and the reference answer.

#Rubric
A helpful and relevant response:
- Directly and completely answers the core question
- Contains accurate, complete, and practical information
- Includes contextually relevant details
- Matches or improves upon the reference answer

An unhelpful or irrelevant response:
- Fails to answer the main question
- Omits critical information present in the reference
- Includes unrelated, vague, or misleading content

#Instructions
- Carefully read and understand full meaning (including edge cases) of the user query, assistant response and the reference answer.
- Determine how effectively the assistant response meets the needs of the query.
- Compare it with the reference to assess coverage, relevance, and detail.
- Consider whether it would be helpful to a real user trying to get a useful answer.

#Reminder
- Your evaluation should be based on how useful the assistant response is **relative to the reference**.
- Clearly state strengths and weaknesses.
- Avoid assumptions beyond what is explicitly provided.

User Question: {user_query}
Assistant Response: {model_output}
Reference Answer: {reference_output}

##Return your evaluation in this exact JSON format:

Example 1:
{{
  "score": 3,
  "reasoning": "The assistant partially addressed the question but missed critical context about the structure of neural networks provided in the reference."
}}

Example 2:
{{
  "score": 5,
  "reasoning": "The assistant gave a complete, accurate, and helpful explanation that covers all key details from the reference."
}}"""
)

DEFAULT_RELEVANCE_PROMPT = PromptTemplate(
    """You are an expert data labeler evaluating retrieved context for its relevance to the user question. Score from 1 to 5 using this rubric:

#Rubric
- Score 5: Highly relevant — Directly helps answer the question with accurate and useful information.
- Score 4: Mostly relevant — Contains useful but possibly partial or incomplete info.
- Score 3: Somewhat relevant — Marginally related; offers weak or tangential clues.
- Score 2: Barely relevant — Mostly off-topic, with minor related elements.
- Score 1: Irrelevant — Unrelated, misleading, or contains no useful information.

Examples:
- Score 5: Context includes facts that directly answer the question.
- Score 3: Context mentions related entities but does not help answer.
- Score 1: Context discusses unrelated topics or incorrect facts.

#Instructions
- Read the user question and assistant response.
- Identify what kind of information is needed to answer the question.
- Examine the retrieved context for:
  - Directly helpful information
  - Partially related or tangential information
  - Unrelated or misleading content
- Use the rubric to assign a score.
- Be specific in reasoning: cite which context helped or didn't, and why.

#Output Format
Return your evaluation in JSON:
{{
  "score": <1-5>,
  "reasoning": "<brief explanation of your decision>"
}}

#User Question: {user_query}
#Assistant Response: {model_output}
"""
)

DEFAULT_FACTUALITY_PROMPT = PromptTemplate(
    """You are an expert data labeler evaluating model outputs for correctness. Your task is to assign a score from 1 to 5 based on the following rubric:

#Rubric
  A correct answer:
  - Provides accurate and complete information
  - Contains no factual errors
  - Addresses all parts of the question
  - Is logically consistent
  - Uses precise and accurate terminology

  When scoring, you should penalize:
  - Factual errors or inaccuracies
  - Major factual omissions
  - Incomplete or partial answers
  - Misleading or ambiguous statements
  - Incorrect terminology
  - Logical inconsistencies
  - Missing major key information (DO NOT penalize for minor supplementary omissions)

#Instructions
  - Carefully read the input and output
  - Check for factual accuracy and completeness
  - Focus on correctness of information rather than style or verbosity

#Reminder
  The goal is to evaluate factual correctness and completeness of the response.

#User Question: {user_query}
#Assistant Response: {model_output}
#Reference Answer: {reference_output}

##Provide a JSON response:
{{
  "score": <1-5>,
  "reasoning": "<explanation>"
}}"""
)


DEFAULT_FLUENCY_PROMPT = PromptTemplate(
    """
You are an expert data labeler evaluating assistant's response for fluency and coherence.

#Rubric
  A fluent answer:
  - Uses correct grammar, punctuation, and spelling
  - Reads smoothly and naturally, like native-level writing
  - Has clear and coherent sentence structure
  - Maintains a consistent tone and logical flow
  - Avoids awkward phrasing and unnecessary repetition

  When scoring, you should penalize:
  - Grammatical or punctuation errors
  - Awkward or unnatural phrasing
  - Run-on or fragmented sentences
  - Jarring transitions or incoherent structure
  - Excessive repetition or verbosity that affects readability

#Instructions
  - Carefully read the assistant's output for grammar, clarity, and natural flow
  - Do not judge factual correctness, relevance, or helpfulness — only language quality
  - Consider whether the response would sound natural to a native speaker
  - Identify any specific grammatical or stylistic issues that impact readability

#User: {user_query}
#Assistant: {model_output}

##Respond in JSON:
{{
  "score": <1-5>,
  "reasoning": "<explanation>"
}}
"""
)

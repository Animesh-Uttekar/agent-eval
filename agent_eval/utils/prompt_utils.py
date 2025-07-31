from agent_eval.models.base_wrapper import BaseLLMWrapper


def adapt_prompt_for_model(prompt: str, model_wrapper: BaseLLMWrapper) -> str:
    suffixes = {
        "OpenAIWrapper": "\n\nRespond ONLY with a valid JSON object. Do not include any explanation or prefix.",
        "HuggingFaceWrapper": "\n\nYour response must be STRICTLY valid JSON with no commentary or formatting.",
        "ClaudeWrapper": "\n\nOutput ONLY a single valid JSON object without explanation or prefix/suffix.",
        "MistralWrapper": "\n\nReturn ONLY valid JSON. No extra commentary or wrapping.",
        "OllamaWrapper": "\n\nPlease return a single valid JSON object. No prose or surrounding text.",
    }

    model_name = type(model_wrapper).__name__
    return prompt + suffixes.get(model_name, "")

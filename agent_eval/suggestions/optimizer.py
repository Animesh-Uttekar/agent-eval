from agent_eval.prompts.templates import PromptTemplate
from agent_eval.utils.thresholds import MetricThreshold, CategoryThreshold
from agent_eval.utils.logging_utils import loggable, get_logger
from agent_eval.prompts.optimizer_prompt import IMPROVE_PROMPT_TMPL
from typing import Optional


class PromptOptimizer:
    def __init__(self, llm, task_type=None):
        """
        Args:
            llm: An instance of BaseLLMWrapper or compatible interface that supports .generate(prompt).
        """
        self.llm = llm
        self.task_type = task_type
        self.logger = get_logger()

    @loggable
    def suggest(
        self,
        prompt: str,
        metrics_results: dict,
        judges_results: dict,
        model_output: Optional[str] = None,
        reference_output: Optional[str] = None,
    ) -> dict:
        """
        Suggest an improved prompt based on failed metrics.

        Args:
            prompt (str): The original user prompt.
            evaluation_results (dict): Results of evaluation (metric/judge → result dict).

        Returns:
            dict: {
                "original_prompt": str,
                "improved_prompt": str,
                "feedback_summary": str
            }
        """
        feedback_items = []

        for name, res in metrics_results.items():
            threshold = MetricThreshold.get_threshold(name)
            score = float(res.get("score", 0.0))
            if threshold is not None and score < threshold:
                reason = res.get("reasoning", res.get("suggestion", "")).strip()
                feedback_items.append(
                    f"- **{name}** score {score:.2f} < {threshold:.2f}\n  Reason: {reason or 'No reasoning provided'}"
                )
                
        for name, res in judges_results.items():
            threshold = CategoryThreshold.get_threshold(name)
            score = float(res.get("score", 0.0))
            if threshold is not None and score < threshold:
                reason = res.get("reasoning", res.get("suggestion", "")).strip()
                feedback_items.append(
                    f"- **{name}** score {score:.2f} < {threshold:.2f}\n  Reason: {reason or 'No reasoning provided'}"
                )

        if feedback_items:
            feedback_summary = "\n".join(feedback_items)
            formatted_prompt = self._build_improvement_prompt(
                prompt, feedback_summary, model_output, reference_output
            )

            try:
                improved_prompt = self.llm.generate(formatted_prompt).strip()
                if not improved_prompt or improved_prompt == prompt:
                    self.logger.warning(
                        f"Improved prompt is identical or empty: {improved_prompt}"
                    )
                    improved_prompt = prompt
            except Exception as e:
                self.logger.error(f"Prompt improvement failed: {str(e)}")
                improved_prompt = prompt

            return {
                "original_prompt": prompt,
                "improved_prompt": improved_prompt,
                "feedback_summary": feedback_summary,
            }

        return {
            "original_prompt": prompt,
            "improved_prompt": prompt,
            "feedback_summary": "No issues found. Prompt is fine.",
        }

    def _build_improvement_prompt(
        self,
        system_prompt: str,
        evaluation_feedback: str,
        model_output: Optional[str],
        reference_output: Optional[str],
    ) -> str:
        """
        Format the improve-prompt template with the provided context.
        """
        return IMPROVE_PROMPT_TMPL.format(
            system_prompt=system_prompt,
            reference_output=reference_output,
            model_output=model_output,
            evaluation_feedback=evaluation_feedback,
            task_type=self.task_type.value if self.task_type else "general",
        )

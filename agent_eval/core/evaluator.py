from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union
from functools import lru_cache

from agent_eval.models.wrapper_factory import ModelWrapperFactory
from dotenv.main import logger
from agent_eval.metrics.base import MetricRegistry
from agent_eval.judges.base import JudgeRegistry
from agent_eval.utils.logging_utils import loggable, get_logger
from agent_eval.utils.thresholds import MetricThreshold


class EvaluationResult:
    def __init__(
        self,
        metrics: dict,
        judges: dict,
        prompt: str,
        model_output: str = None,
        reference_output: str = None,
        error: str = None,
    ):
        self.metrics = metrics or {}
        self.judges = judges or {}
        self.prompt = prompt
        self.model_output = model_output
        self.reference_output = reference_output
        self.error = error

    def to_dict(self) -> dict:
        """
        Return a structured (nested) dictionary result.
        """
        result = {
            "metrics": self.metrics,
            "judges": self.judges,
            "original_prompt": self.prompt,
            "model_output": self.model_output,
            "reference_output": self.reference_output,
        }

        if self.error is not None:
            result["error"] = self.error

        return result


class Evaluator:
    def __init__(
        self,
        model=None,
        model_name=None,
        model_provider=None,
        metrics=None,
        judges=None,
        metric_workers=4,
        judge_workers=4,
    ):
        self.metrics = metrics or []
        self.judges = judges or []
        self.metric_workers = metric_workers
        self.judge_workers = judge_workers
        self.model = (
            ModelWrapperFactory.wrap(
                model, model_name=model_name, provider=model_provider
            )
            if model
            else None
        )

    @loggable
    def _resolve_metrics(self, metric_names_or_category: Union[str, list]) -> list:
        resolved = []
        if not metric_names_or_category:
            return resolved
        if isinstance(metric_names_or_category, str):
            metric_names_or_category = [metric_names_or_category]
        for item in metric_names_or_category:
            metric_cls = get_metric_by_name(item)
            if metric_cls:
                resolved.append(metric_cls())
            else:
                logger.warning(f"Metric '{item}' not resolved.")
        return resolved

    @loggable
    def _resolve_judges(self, judge_names_or_category: Union[str, list], model) -> list:
        resolved = []
        if not judge_names_or_category:
            return resolved
        if isinstance(judge_names_or_category, str):
            judge_names_or_category = [judge_names_or_category]
        for item in judge_names_or_category:
            judge_cls = get_judge_by_name(item)
            if judge_cls:
                resolved.append(judge_cls(model=model))
            else:
                logger.warning(f"Judge '{item}' not resolved.")
        return resolved

    def _evaluate_metrics(self, metrics, model_output, reference_output):
        results = {}
        with ThreadPoolExecutor(max_workers=self.metric_workers) as executor:
            futures = {
                executor.submit(metric.evaluate, model_output, reference_output): metric
                for metric in metrics
            }
            for future in as_completed(futures):
                metric = futures[future]
                name = metric.criteria
                try:
                    results[name] = future.result()
                except Exception as e:
                    results[name] = {"score": 0.0, "suggestion": "", "error": str(e)}
        return results

    def _evaluate_judges(self, judges, prompt, model_output, reference_output):
        results = {}
        with ThreadPoolExecutor(max_workers=self.judge_workers) as executor:
            futures = {
                executor.submit(
                    judge.judge, prompt, model_output, reference_output
                ): judge
                for judge in judges
            }
            for future in as_completed(futures):
                judge = futures[future]
                judge_criteria = judge.criteria
                try:
                    results[judge_criteria] = future.result()
                except Exception as e:
                    results[judge_criteria] = {
                        "score": 0.0,
                        "reasoning": "",
                        "error": str(e),
                    }
        return results

    @loggable
    def _generate_with_model(self, model, prompt, **kwargs):
        try:
            if hasattr(model, "generate") and (
                not hasattr(model, "chat") or not hasattr(model.chat, "completions")
            ):
                response = model.generate(prompt, **kwargs)
                return response[0] if isinstance(response, list) else response
            elif hasattr(model, "chat") and hasattr(model.chat, "completions"):
                response = model.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}], **kwargs
                )
                return response.choices[0].message.content
            elif hasattr(model, "complete"):
                return model.complete(prompt, **kwargs)
            elif callable(model):
                response = model(prompt, **kwargs)
                if isinstance(response, dict) and "text" in response:
                    return response["text"]
                elif isinstance(response, str):
                    return response
                else:
                    return str(response)
            else:
                raise ValueError(f"Unsupported model type: {type(model)}")
        except Exception as e:
            raise Exception(f"Model generation failed: {str(e)}")

    @loggable
    def _generate_result(
        self,
        metrics_results,
        judges_results,
        prompt,
        model_output,
        reference_output,
        error=None,
    ):
        metrics = {k: v for k, v in metrics_results.items()}
        judges = {k: v for k, v in judges_results.items()}

        return EvaluationResult(
            metrics=metrics,
            judges=judges,
            prompt=prompt,
            model_output=model_output,
            reference_output=reference_output,
            error=error,
        ).to_dict()

    @loggable
    def evaluate(
        self,
        prompt,
        model_output=None,
        reference_output=None,
        metrics=None,
        judges=None,
        model=None,
        **model_kwargs,
    ):
        logger = get_logger()
        logger.info(
            f"Starting evaluation: metrics_override={metrics!r}, judges_override={judges!r}"
        )

        model = model or self.model

        try:
            if model is not None:
                if model_output is None:
                    model_output = self._generate_with_model(
                        model, prompt, **model_kwargs
                    )
            elif model is None and judges is not None and len(judges) > 0:
                raise ValueError(
                    "'model' must be provided with evaluate with llm judges"
                )
            elif model is None and metrics:
                raise ValueError("Either 'model' or 'model_output' must be provided")

            results = {}
            metrics_to_run = self._resolve_metrics(metrics) if metrics else self.metrics
            metrics_results = self._evaluate_metrics(
                metrics_to_run, model_output, reference_output
            )

            judges_to_run = (
                self._resolve_judges(judges, model) if judges else self.judges
            )
            judges_results = self._evaluate_judges(
                judges_to_run, prompt, model_output, reference_output
            )

            results = self._generate_result(
                metrics_results,
                judges_results,
                prompt,
                model_output,
                reference_output,
            )

            return results

        except Exception as e:
            logger.exception("Evaluation failed")
            return self._generate_result(
                {},
                {},
                prompt,
                None,
                model_output,
                reference_output,
                error=str(e),
            )


@lru_cache(maxsize=None)
def get_metric_by_name(name: str):
    return MetricRegistry.get_by_name(name)


@lru_cache(maxsize=None)
def get_judge_by_name(name: str):
    return JudgeRegistry.get_by_name(name)

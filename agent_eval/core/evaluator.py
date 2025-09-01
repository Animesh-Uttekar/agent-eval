from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, Optional, List, Dict, Any, Callable
from functools import lru_cache
import inspect
import uuid

from agent_eval.utils.model_utils import wrap_model_safely
from dotenv.main import logger
from agent_eval.metrics.base import MetricRegistry
from agent_eval.judges.base import JudgeRegistry
from agent_eval.suggestions.optimizer import PromptOptimizer
from agent_eval.utils.logging_utils import loggable, get_logger
from agent_eval.utils.thresholds import MetricThreshold
from agent_eval.core.cache import cache_instance


class EnhancedEvaluationResult:
    """
    Enhanced evaluation result with comprehensive insights and recommendations.
    """
    
    def __init__(self, result_data: Dict[str, Any]):
        self._data = result_data
        
    @property
    def overall_score(self) -> float:
        """Overall evaluation score (0-1)."""
        # Calculate weighted average of all scores
        scores = []
        
        # Include metric scores
        for metric_name, metric_result in self._data.get("metrics", {}).items():
            if isinstance(metric_result, dict) and "score" in metric_result:
                if metric_result["score"] is not None:
                    scores.append(metric_result["score"])
        
        # Include judge scores  
        for judge_name, judge_result in self._data.get("judges", {}).items():
            if isinstance(judge_result, dict) and "score" in judge_result:
                if judge_result["score"] is not None:
                    scores.append(judge_result["score"])
        
        return sum(scores) / len(scores) if scores else 0.0
    
    @property
    def improvement_suggestions(self) -> List[str]:
        """Specific, actionable improvement suggestions."""
        suggestions = []
        
        # Collect suggestions from judges
        for judge_result in self._data.get("judges", {}).values():
            if isinstance(judge_result, dict):
                judge_suggestions = judge_result.get("improvement_suggestions", [])
                suggestions.extend(judge_suggestions)
        
        # Add domain-specific suggestions
        domain_result = self._data.get("domain_evaluation", {})
        domain_suggestions = domain_result.get("improvement_recommendations", [])
        suggestions.extend(domain_suggestions)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions
    
    @property
    def performance_insights(self) -> Dict[str, Any]:
        """Performance and optimization insights."""
        return {
            "overall_score": self.overall_score,
            "cache_performance": self._data.get("cache_info", {}),
            "domain_compliance": self._data.get("domain_evaluation", {}),
            "bias_analysis": self._data.get("bias_analysis", {}),
            "meta_evaluation": self._data.get("meta_evaluation", {})
        }
    
    @property
    def passes_quality_gates(self) -> bool:
        """Whether the evaluation passes all quality gates."""
        # Check domain-specific requirements
        domain_result = self._data.get("domain_evaluation", {})
        if domain_result and domain_result.get("compliance_score", 1.0) < 0.8:
            return False
        
        # Check overall score threshold
        if self.overall_score < 0.7:
            return False
            
        return True
    
    @property 
    def detailed_breakdown(self) -> Dict[str, Any]:
        """Complete detailed breakdown of all evaluation components."""
        return self._data
    
    def __getattr__(self, name):
        """Allow direct access to underlying data."""
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __getitem__(self, key):
        """Allow dictionary-style access for backward compatibility."""
        return self._data[key]
    
    def __contains__(self, key):
        """Support 'in' operator for backward compatibility."""
        return key in self._data
    
    def get(self, key, default=None):
        """Dictionary-style get method for backward compatibility."""
        return self._data.get(key, default)
    
    def keys(self):
        """Dictionary-style keys method for backward compatibility."""
        return self._data.keys()
    
    def values(self):
        """Dictionary-style values method for backward compatibility."""
        return self._data.values()
    
    def items(self):
        """Dictionary-style items method for backward compatibility."""
        return self._data.items()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return self._data


class EvaluationResult:
    def __init__(
        self,
        metrics: dict,
        judges: dict,
        prompt: str,
        improved_prompt: str = None,
        model_output: str = None,
        reference_output: str = None,
        user_query: str = None,
        attempts: int = 0,
        error: str = None,
    ):
        self.metrics = metrics or {}
        self.judges = judges or {}
        self.prompt = prompt
        self.improved_prompt = improved_prompt
        self.model_output = model_output
        self.reference_output = reference_output
        self.user_query = user_query
        self.attempts = attempts
        self.error = error

    def to_dict(self) -> dict:
        """
        Return a structured (nested) dictionary result.
        """
        result = {
            "metrics": self.metrics,
            "judges": self.judges,
            "original_prompt": self.prompt,
            "improved_prompt": self.improved_prompt,
            "model_output": self.model_output,
            "reference_output": self.reference_output,
            "user_query": self.user_query,
            "attempts": self.attempts,
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
        prompt_optimizer=False,
        max_prompt_improvements=1,
        domain=None,
        auto_generate_tests=True,
        enable_caching=True,
        cache_dir=".agent_eval_cache",
        quality_gates=None,
    ):
        self.metrics = metrics or []
        self.judges = judges or []
        self.metric_workers = metric_workers
        self.judge_workers = judge_workers
        
        # Store model info (needed for backward compatibility)
        self.model_name = model_name
        self.model_provider = model_provider
        
        # Use shared model wrapping utility
        self.model = wrap_model_safely(model, model_name, model_provider)
                    
        self.prompt_optimizer = prompt_optimizer
        self.max_prompt_improvements = max_prompt_improvements
        
        # Enhanced features
        self.domain = domain
        self.auto_generate_tests = auto_generate_tests
        self.quality_gates = quality_gates or {}

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

    def _evaluate_metrics(
        self, metrics, model_output, reference_output, prompt, user_query
    ):
        results = {}
        with ThreadPoolExecutor(max_workers=self.metric_workers) as executor:
            futures = {
                executor.submit(
                    metric.evaluate, model_output, reference_output, prompt, user_query
                ): metric
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
                # Use first alias as key for backward compatibility (e.g., "factuality")
                judge_key = judge.aliases[0] if hasattr(judge, 'aliases') and judge.aliases else judge.criteria
                try:
                    results[judge_key] = future.result()
                except Exception as e:
                    results[judge_key] = {
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
        original_prompt,
        prompt,
        model_output,
        reference_output,
        user_query,
        attempts,
        error=None,
    ):
        metrics = {k: v for k, v in metrics_results.items()}
        judges = {k: v for k, v in judges_results.items()}

        return EvaluationResult(
            metrics=metrics,
            judges=judges,
            prompt=original_prompt,
            improved_prompt=prompt if prompt != original_prompt else None,
            model_output=model_output,
            reference_output=reference_output,
            user_query=user_query,
            attempts=attempts,
            error=error,
        ).to_dict()

    @loggable
    def evaluate(
        self,
        prompt,
        model_output=None,
        reference_output=None,
        user_query=None,
        metrics=None,
        judges=None,
        model=None,
        prompt_optimizer=False,
        max_prompt_improvements=1,
        async_mode=False,
        # NEW: Intelligent test generation parameters
        generate_testcases=False,
        num_scenarios=50,
        test_dataset=None,
        unit_test_cases=None,
        # NEW: User examples for learning
        user_test_cases=None,
        sample_interactions=None,
        analyze_behavior=True,
        **model_kwargs,
    ):
        logger = get_logger()
        model = model or self.model
        original_prompt = prompt
        attempts = 0

        # AUTOMATIC TEST GENERATION: Handle intelligent test case generation
        if generate_testcases:
            logger.info("Generating intelligent test scenarios automatically...")
            
            # REGISTRY-BASED INTELLIGENT SELECTION: Uses existing metrics/judges
            from agent_eval.core.intelligent_selector import IntelligentSelector
            
            logger.info("Using intelligent registry-based selection (most cost-effective approach)...")
            selector = IntelligentSelector(model, self.model_name or "gpt-3.5-turbo")
            
            # Step 1: Analyze agent and intelligently select from registry (1 LLM call)
            logger.info("Analyzing agent and selecting optimal metrics/judges from registry...")
            selection_result = selector.analyze_and_select_optimal(
                agent_description=prompt,
                sample_output=model_output,
                max_metrics=4,
                max_judges=3
            )
            
            evaluation_plan = selector.create_evaluation_plan(selection_result)
            
            logger.info(f"Domain: {selection_result.get('agent_analysis', {}).get('domain', 'Unknown')}")
            logger.info(f"Selected {evaluation_plan['metrics_plan']['count']} optimal metrics from registry")
            logger.info(f"Selected {evaluation_plan['judges_plan']['count']} optimal judges from registry")
            logger.info(f"Generated {len(selection_result.get('test_scenarios', []))} test scenarios")
            
            # Create agent function for real testing
            def agent_function(test_prompt: str) -> str:
                """Wrapper to make model callable for testing."""
                if hasattr(model, 'chat') and hasattr(model.chat, 'completions'):
                    create_kwargs = {"messages": [{"role": "user", "content": test_prompt}]}
                    if self.model_name:
                        create_kwargs["model"] = self.model_name
                    create_kwargs.update(model_kwargs)
                    response = model.chat.completions.create(**create_kwargs)
                    return response.choices[0].message.content
                else:
                    return self.model.generate(test_prompt, **model_kwargs)
            
            # Step 2: Run test scenarios with real agent testing (1 call per scenario)
            logger.info("Running test scenarios...")
            scenario_results = []
            
            for scenario in selection_result.get('test_scenarios', []):
                try:
                    agent_response = agent_function(scenario.get('prompt', ''))
                    scenario_results.append({
                        'name': scenario.get('name', 'Unknown'),
                        'prompt': scenario.get('prompt', ''),
                        'agent_response': agent_response,
                        'focus': scenario.get('focus', 'General testing')
                    })
                except Exception as e:
                    logger.warning(f"Scenario {scenario.get('name', 'Unknown')} failed: {e}")
            
            logger.info(f"Completed {len(scenario_results)} test scenarios")
            
            # Step 3: Evaluate using selected metrics and judges
            logger.info("Evaluating using selected registry-based metrics and judges...")
            
            # Run metrics evaluation
            metrics_results = {}
            for metric_info in evaluation_plan['metrics_plan']['instances']:
                metric_name = metric_info['name']
                metric_instance = metric_info['instance']
                requires_ref = metric_info.get('requires_reference', True)
                
                if requires_ref and not reference_output:
                    logger.info(f"Skipping {metric_name} - requires reference output")
                    continue
                
                try:
                    # Evaluate across all scenarios
                    metric_scores = []
                    for scenario in scenario_results:
                        if requires_ref:
                            score = metric_instance.evaluate(
                                scenario['agent_response'], 
                                reference_output,
                                scenario['prompt']
                            )
                        else:
                            score = metric_instance.evaluate(
                                scenario['agent_response'],
                                prompt=scenario['prompt']
                            )
                        metric_scores.append(score)
                    
                    avg_score = sum(metric_scores) / len(metric_scores) if metric_scores else 0.0
                    metrics_results[metric_name] = {
                        'score': avg_score,
                        'individual_scores': metric_scores,
                        'rationale': metric_info.get('rationale', '')
                    }
                    
                except Exception as e:
                    logger.warning(f"Metric {metric_name} evaluation failed: {e}")
                    metrics_results[metric_name] = {
                        'score': 0.0,
                        'error': str(e),
                        'rationale': metric_info.get('rationale', '')
                    }
            
            # Run judges evaluation (1 batched LLM call)
            judges_results = self._evaluate_with_selected_judges(
                evaluation_plan['judges_plan']['classes'],
                scenario_results,
                model,
                **model_kwargs
            )
            
            # Calculate overall score
            all_scores = []
            all_scores.extend([r['score'] for r in metrics_results.values() if 'score' in r])
            all_scores.extend([r.get('score', 0.0) for r in judges_results.values()])
            overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
            
            # Format final results
            aggregated_result = {
                'overall_score': overall_score,
                'scenario_results': [
                    {
                        'name': r.get('name', 'Unknown'),
                        'type': 'registry_selected',
                        'score': overall_score,  # Could calculate per-scenario if needed
                        'passed': overall_score > 0.6
                    } for r in scenario_results
                ],
                'metrics': metrics_results,
                'judges': judges_results,
                'agent_analysis': selection_result.get('agent_analysis', {}),
                'selected_from_registry': {
                    'metrics': [m['name'] for m in evaluation_plan['metrics_plan']['instances']],
                    'judges': [j['name'] for j in evaluation_plan['judges_plan']['classes']]
                },
                'evaluation_approach': 'Registry-based intelligent selection',
                'cost_optimization': f"Used {1 + len(scenario_results) + 1} LLM calls vs 40+",
                'original_prompt': original_prompt,
                'model_output': model_output
            }
            
            logger.info(f"Evaluation complete! Overall Score: {overall_score:.3f}")
            logger.info(f"Total Cost: ~{2 + len(scenario_results)} LLM calls (ultra cost-effective!)")
            
            return EnhancedEvaluationResult(aggregated_result)

        # Handle custom unit test cases
        if unit_test_cases:
            logger.info(f"Running {len(unit_test_cases)} custom unit test cases...")
            return self._evaluate_unit_test_cases(
                unit_test_cases, model, metrics, judges, prompt_optimizer, 
                max_prompt_improvements, **model_kwargs
            )
        
        # Handle custom dataset evaluation
        if test_dataset:
            logger.info(f"Running evaluation on dataset with {len(test_dataset)} entries...")
            return self._evaluate_dataset(
                test_dataset, model, metrics, judges, prompt_optimizer,
                max_prompt_improvements, **model_kwargs
            )

        if async_mode:
            import asyncio
            from agent_eval.core.async_evaluator import AsyncEvaluator

            logger.info("Delegating to AsyncEvaluator via asyncio.run()...")
            async_evaluator = AsyncEvaluator(self)
            return asyncio.run(
                async_evaluator.evaluate_async(
                    prompt=prompt,
                    model_output=model_output,
                    reference_output=reference_output,
                    user_query=user_query,
                    metrics=metrics,
                    judges=judges,
                    model=model,
                    prompt_optimizer=prompt_optimizer,
                    max_prompt_improvements=max_prompt_improvements,
                    model_kwargs=model_kwargs,
                )
            )

        cached_result = cache_instance.get(
            prompt=prompt,
            model_output=model_output,
            reference_output=reference_output,
            user_query=user_query,
            metrics=metrics or [m.criteria for m in self.metrics],
            judges=judges or [j.criteria for j in self.judges],
            prompt_optimizer=prompt_optimizer,
            max_prompt_improvements=max_prompt_improvements,
        )
        if cached_result:
            logger.info("Returning cached evaluation result")
            return EnhancedEvaluationResult(cached_result)

        while attempts < max_prompt_improvements:
            try:
                if model is not None and (model_output is None or attempts > 0):
                    model_output = self._generate_with_model(
                        model, prompt, **model_kwargs
                    )

                if model is None and judges:
                    raise ValueError(
                        "'model' must be provided with evaluate with llm judges"
                    )
                if model is None and metrics:
                    raise ValueError(
                        "Either 'model' or 'model_output' must be provided"
                    )

                metrics_to_run = (
                    self._resolve_metrics(metrics) if metrics else self.metrics
                )
                metrics_results = self._evaluate_metrics(
                    metrics_to_run, model_output, reference_output, prompt, user_query
                )

                judges_to_run = (
                    self._resolve_judges(judges, model) if judges else self.judges
                )
                judges_results = self._evaluate_judges(
                    judges_to_run, prompt, model_output, reference_output
                )

                results = self._generate_result(
                    metrics_results=metrics_results,
                    judges_results=judges_results,
                    original_prompt=original_prompt,
                    prompt=prompt,
                    model_output=model_output,
                    reference_output=reference_output,
                    user_query=user_query,
                    attempts=attempts,
                )

                cache_instance.set(
                    result=results,
                    prompt=prompt,
                    model_output=model_output,
                    reference_output=reference_output,
                    user_query=user_query,
                    metrics=metrics or [m.criteria for m in self.metrics],
                    judges=judges or [j.criteria for j in self.judges],
                    prompt_optimizer=prompt_optimizer,
                    max_prompt_improvements=max_prompt_improvements,
                )

                if not prompt_optimizer:
                    return EnhancedEvaluationResult(results)

                optimizer = PromptOptimizer(model)
                opt_result = optimizer.suggest(
                    prompt=prompt,
                    metrics_results=metrics_results,
                    judges_results=judges_results,
                    model_output=model_output,
                    reference_output=reference_output,
                )
                new_prompt = opt_result.get("improved_prompt", prompt)

                if new_prompt == prompt:
                    return EnhancedEvaluationResult(results)

                prompt = new_prompt
                attempts += 1
                logger.info(
                    f"Re-running evaluation with improved prompt (attempt {attempts})"
                )

            except Exception as e:
                logger.exception("Evaluation failed")
                return EnhancedEvaluationResult(self._generate_result(
                    {},
                    {},
                    original_prompt=original_prompt,
                    prompt=None,
                    model_output=model_output,
                    reference_output=reference_output,
                    user_query=user_query,
                    attempts=attempts,
                    error=str(e),
                ))

    def _analyze_agent_from_callable(self, model) -> str:
        """Analyze agent characteristics from the callable model."""
        analysis_parts = []
        
        if hasattr(model, '__name__'):
            analysis_parts.append(f"Agent name: {model.__name__}")
        
        if hasattr(model, '__doc__') and model.__doc__:
            analysis_parts.append(f"Description: {model.__doc__}")
            
        if hasattr(model, 'description'):
            analysis_parts.append(f"Description: {model.description}")
        
        # Check for common agent patterns in code
        if hasattr(model, '__call__'):
            try:
                signature = inspect.signature(model.__call__)
                analysis_parts.append(f"Callable with signature: {signature}")
            except:
                pass
        
        return " ".join(analysis_parts) if analysis_parts else "AI agent for general tasks"

    def _evaluate_single(self, prompt, model_output=None, reference_output=None, 
                        user_query=None, metrics=None, judges=None, model=None,
                        prompt_optimizer=False, max_prompt_improvements=1, **model_kwargs):
        """Evaluate a single prompt-output pair (used internally for scenarios)."""
        # This is the original evaluate logic without test generation
        model = model or self.model
        original_prompt = prompt
        attempts = 0
        
        # Skip cache for internal calls
        while attempts <= max_prompt_improvements:
            current_prompt = prompt
            attempts += 1
            
            if model_output is None and model:
                wrapped_model = wrap_model_safely(model, self.model_name)
                model_output = wrapped_model.generate(current_prompt, **model_kwargs)
            
            # Run metrics
            metric_results = {}
            if metrics:
                for metric_name in metrics:
                    metric = get_metric_by_name(metric_name)
                    if metric and reference_output:
                        try:
                            score = metric.compute(model_output, reference_output)
                            metric_results[metric_name] = {
                                "score": score,
                                "suggestion": metric.get_improvement_suggestion(score)
                            }
                        except Exception as e:
                            logger.warning(f"Metric {metric_name} failed: {e}")
                            metric_results[metric_name] = {"score": None, "error": str(e)}
            
            # Run judges
            judge_results = {}
            if judges:
                for judge_name in judges:
                    judge = get_judge_by_name(judge_name)
                    if judge:
                        try:
                            judge_result = judge.evaluate(
                                prompt=current_prompt,
                                response=model_output,
                                reference=reference_output,
                                user_query=user_query or current_prompt,
                                model=model,
                                **model_kwargs
                            )
                            judge_results[judge_name] = judge_result
                        except Exception as e:
                            logger.warning(f"Judge {judge_name} failed: {e}")
                            judge_results[judge_name] = {"score": None, "error": str(e)}
            
            # Check if we should optimize prompt
            if prompt_optimizer and attempts <= max_prompt_improvements:
                scores = self._extract_scores(metric_results, judge_results)
                if any(score < 0.7 for score in scores if score is not None):
                    optimizer = PromptOptimizer(model=model, model_name=self.model_name)
                    improved_prompt = optimizer.optimize_prompt(
                        current_prompt, metric_results, judge_results
                    )
                    if improved_prompt != current_prompt:
                        prompt = improved_prompt
                        model_output = None  # Re-generate with improved prompt
                        continue
            
            break
        
        return {
            "metrics": metric_results,
            "judges": judge_results,
            "original_prompt": original_prompt,
            "model_output": model_output,
            "reference_output": reference_output or "",
            "attempts": attempts
        }

    def _aggregate_scenario_results(self, scenario_results: List[Dict], 
                                  behavior_profile, original_prompt: str, 
                                  original_output: str) -> Dict[str, Any]:
        """Aggregate results from multiple test scenarios."""
        
        # Calculate aggregate scores
        all_scores = []
        scenario_summaries = []
        
        for result in scenario_results:
            # Extract scores from this scenario
            scenario_scores = []
            
            # Add metric scores
            for metric_name, metric_data in result.get("metrics", {}).items():
                if isinstance(metric_data, dict) and metric_data.get("score") is not None:
                    scenario_scores.append(metric_data["score"])
            
            # Add judge scores
            for judge_name, judge_data in result.get("judges", {}).items():
                if isinstance(judge_data, dict) and judge_data.get("score") is not None:
                    scenario_scores.append(judge_data["score"])
            
            scenario_avg = sum(scenario_scores) / len(scenario_scores) if scenario_scores else 0.0
            all_scores.append(scenario_avg)
            
            # Create scenario summary
            scenario_summaries.append({
                "name": result.get("scenario_name", "Unknown"),
                "type": result.get("scenario_type", "unknown"),
                "score": scenario_avg,
                "prompt": result.get("original_prompt", ""),
                "output": result.get("model_output", ""),
                "expected_quality": result.get("expected_quality", {}),
                "min_score": result.get("min_score", 0.0),
                "passed": scenario_avg >= result.get("min_score", 0.0),
                "tags": result.get("tags", []),
                "metrics": result.get("metrics", {}),
                "judges": result.get("judges", {})
            })
        
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        # Performance analysis
        passed_scenarios = [s for s in scenario_summaries if s["passed"]]
        failed_scenarios = [s for s in scenario_summaries if not s["passed"]]
        
        performance_insights = {
            "total_scenarios": len(scenario_summaries),
            "passed_scenarios": len(passed_scenarios),
            "failed_scenarios": len(failed_scenarios),
            "pass_rate": len(passed_scenarios) / len(scenario_summaries) if scenario_summaries else 0.0,
            "average_score": overall_score,
            "scenario_breakdown": {
                scenario_type: len([s for s in scenario_summaries if s["type"] == scenario_type])
                for scenario_type in set(s["type"] for s in scenario_summaries)
            }
        }
        
        # Actionable insights
        actionable_insights = []
        
        if behavior_profile:
            if behavior_profile.weakness_areas:
                actionable_insights.append(f"Focus on improving: {', '.join(behavior_profile.weakness_areas)}")
            
            if behavior_profile.custom_metrics_needed:
                actionable_insights.append(f"Consider implementing custom metrics: {', '.join(behavior_profile.custom_metrics_needed)}")
            
            if performance_insights["pass_rate"] < 0.8:
                actionable_insights.append("Pass rate below 80% - review failed scenarios for improvement opportunities")
        
        # Detailed results
        aggregated_result = {
            "overall_score": overall_score,
            "performance_insights": performance_insights,
            "actionable_insights": actionable_insights,
            "scenario_results": scenario_summaries,
            "behavior_profile": behavior_profile.__dict__ if behavior_profile else None,
            "original_prompt": original_prompt,
            "model_output": original_output,
            "reference_output": "",
            "domain_detected": self.domain,
            "test_generation_enabled": True,
            "num_scenarios_generated": len(scenario_summaries)
        }
        
        return aggregated_result

    def _evaluate_unit_test_cases(self, unit_test_cases: List[Dict], model, 
                                 metrics, judges, prompt_optimizer, 
                                 max_prompt_improvements, **model_kwargs) -> EnhancedEvaluationResult:
        """Evaluate custom unit test cases provided by developer."""
        
        results = []
        
        for i, test_case in enumerate(unit_test_cases):
            logger.info(f"Running unit test {i+1}/{len(unit_test_cases)}")
            
            test_prompt = test_case.get("prompt", test_case.get("input", ""))
            expected_output = test_case.get("expected_output", test_case.get("expected", ""))
            test_name = test_case.get("name", f"Unit Test {i+1}")
            
            result = self._evaluate_single(
                prompt=test_prompt,
                model_output=None,  # Generate fresh
                reference_output=expected_output,
                metrics=metrics,
                judges=judges,
                model=model,
                prompt_optimizer=prompt_optimizer,
                max_prompt_improvements=max_prompt_improvements,
                **model_kwargs
            )
            
            result["test_name"] = test_name
            result["test_index"] = i
            results.append(result)
        
        # Aggregate unit test results
        aggregated = self._aggregate_unit_test_results(results)
        return EnhancedEvaluationResult(aggregated)

    def _evaluate_dataset(self, test_dataset: List[Dict], model, metrics, judges,
                         prompt_optimizer, max_prompt_improvements, **model_kwargs) -> EnhancedEvaluationResult:
        """Evaluate on a custom dataset."""
        
        results = []
        
        for i, data_entry in enumerate(test_dataset):
            logger.info(f"Evaluating dataset entry {i+1}/{len(test_dataset)}")
            
            prompt = data_entry.get("prompt", data_entry.get("input", ""))
            reference = data_entry.get("reference", data_entry.get("expected_output", ""))
            
            result = self._evaluate_single(
                prompt=prompt,
                model_output=None,
                reference_output=reference,
                metrics=metrics,
                judges=judges,
                model=model,
                prompt_optimizer=prompt_optimizer,
                max_prompt_improvements=max_prompt_improvements,
                **model_kwargs
            )
            
            result["dataset_index"] = i
            results.append(result)
        
        # Aggregate dataset results  
        aggregated = self._aggregate_dataset_results(results)
        return EnhancedEvaluationResult(aggregated)

    def _aggregate_unit_test_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results from unit test cases."""
        
        all_scores = []
        passed_tests = 0
        
        for result in results:
            scores = self._extract_scores(result.get("metrics", {}), result.get("judges", {}))
            if scores:
                avg_score = sum(scores) / len(scores)
                all_scores.append(avg_score)
                if avg_score >= 0.7:  # Default passing threshold
                    passed_tests += 1
        
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        return {
            "overall_score": overall_score,
            "unit_test_results": results,
            "total_tests": len(results),
            "passed_tests": passed_tests,
            "pass_rate": passed_tests / len(results) if results else 0.0,
            "test_generation_enabled": False,
            "evaluation_mode": "unit_tests"
        }

    def _aggregate_dataset_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results from dataset evaluation."""
        
        all_scores = []
        
        for result in results:
            scores = self._extract_scores(result.get("metrics", {}), result.get("judges", {}))
            if scores:
                avg_score = sum(scores) / len(scores)
                all_scores.append(avg_score)
        
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        return {
            "overall_score": overall_score,
            "dataset_results": results,
            "total_entries": len(results),
            "average_score": overall_score,
            "test_generation_enabled": False,
            "evaluation_mode": "dataset"
        }

    def _extract_scores(self, metric_results: Dict, judge_results: Dict) -> List[float]:
        """Extract numerical scores from metric and judge results."""
        scores = []
        
        for metric_data in metric_results.values():
            if isinstance(metric_data, dict) and metric_data.get("score") is not None:
                scores.append(metric_data["score"])
        
        for judge_data in judge_results.values():
            if isinstance(judge_data, dict) and judge_data.get("score") is not None:
                scores.append(judge_data["score"])
        
        return scores

    def evaluate_agent(
        self,
        agent_function: Callable[[str], str],
        num_test_scenarios: int = 25,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Comprehensive agent evaluation with auto-generated test scenarios.
        
        Args:
            agent_function: Function that takes prompt and returns response
            num_test_scenarios: Number of test scenarios to generate
            **kwargs: Additional arguments
            
        Returns:
            Comprehensive evaluation results across all scenarios
        """
        
        if not self.auto_generate_tests:
            raise ValueError("auto_generate_tests must be True to use evaluate_agent")
        
        # Import here to avoid circular imports
        from agent_eval.test_generation.legacy_generator import IntelligentTestGenerator
        from agent_eval.core.local_engine import LocalEvaluationEngine, EvaluationRequest
        
        # Initialize components
        test_generator = IntelligentTestGenerator()
        
        # Generate comprehensive test scenarios
        agent_description = self._analyze_agent(agent_function)
        domain_value = self.domain if isinstance(self.domain, str) else "general"
        
        test_scenarios = test_generator.generate_comprehensive_test_suite(
            agent_description=agent_description,
            domain=domain_value,
            num_scenarios=num_test_scenarios
        )
        
        # Create evaluation requests
        results = {}
        all_scores = []
        failed_scenarios = []
        
        for i, scenario in enumerate(test_scenarios):
            # Generate agent response
            agent_output = agent_function(scenario.prompt)
            
            # Evaluate this scenario
            result = self.evaluate(
                prompt=scenario.prompt,
                model_output=agent_output,
                reference_output=None,
                metrics=["bleu"],
                judges=["factuality", "relevance", "helpfulness"],
                prompt_optimizer=True,
                max_prompt_improvements=2
            )
            
            results[f"scenario_{i}"] = result
            
            # Calculate scenario score
            scores = []
            if "metrics" in result:
                for metric_result in result["metrics"].values():
                    if isinstance(metric_result, dict) and "score" in metric_result:
                        if metric_result["score"] is not None:
                            scores.append(metric_result["score"])
            
            if "judges" in result:
                for judge_result in result["judges"].values():
                    if isinstance(judge_result, dict) and "score" in judge_result:
                        if judge_result["score"] is not None:
                            scores.append(judge_result["score"])
            
            scenario_score = sum(scores) / len(scores) if scores else 0.0
            all_scores.append(scenario_score)
            
            # Check quality gates
            if scenario_score < scenario.min_score:
                failed_scenarios.append({
                    "scenario": scenario.name,
                    "score": scenario_score,
                    "min_score": scenario.min_score,
                    "type": scenario.scenario_type.value
                })
        
        # Calculate overall metrics
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        pass_rate = (len(all_scores) - len(failed_scenarios)) / len(all_scores) if all_scores else 0.0
        
        return {
            "overall_score": overall_score,
            "pass_rate": pass_rate,
            "total_scenarios": len(test_scenarios),
            "passed_scenarios": len(all_scores) - len(failed_scenarios),
            "failed_scenarios": failed_scenarios,
            "individual_results": results,
            "passes_quality_gates": len(failed_scenarios) == 0
        }
    
    def _analyze_agent(self, model: Any) -> str:
        """Analyze agent to understand its purpose and capabilities."""
        
        # Try to extract information from model
        analysis_parts = []
        
        if hasattr(model, '__name__'):
            analysis_parts.append(f"Agent name: {model.__name__}")
        
        if hasattr(model, '__doc__') and model.__doc__:
            analysis_parts.append(f"Description: {model.__doc__}")
            
        if hasattr(model, 'description'):
            analysis_parts.append(f"Description: {model.description}")
        
        # Check for common agent patterns in code
        if hasattr(model, '__call__'):
            try:
                signature = inspect.signature(model.__call__)
                analysis_parts.append(f"Callable with signature: {signature}")
            except:
                pass
        
        return " ".join(analysis_parts) if analysis_parts else "AI agent for general tasks"
    
    def _evaluate_with_selected_judges(self, judge_classes: List[Dict], scenario_results: List[Dict], 
                                     model, **model_kwargs) -> Dict[str, Any]:
        """Evaluate using selected judges from registry."""
        judges_results = {}
        
        for judge_info in judge_classes:
            judge_name = judge_info['name']
            judge_class = judge_info['class']
            
            try:
                # Instantiate judge with model
                judge_instance = judge_class(model, provider=self.model_provider)
                
                # Evaluate across all scenarios
                judge_scores = []
                judge_reasoning = []
                
                for scenario in scenario_results:
                    try:
                        judge_result = judge_instance.judge(
                            prompt=scenario['prompt'],
                            model_output=scenario['agent_response'],
                            reference_output=None
                        )
                        
                        if isinstance(judge_result, dict):
                            score = judge_result.get('score', 0.0)
                            reasoning = judge_result.get('reasoning', '')
                        else:
                            score = float(judge_result) if judge_result else 0.0
                            reasoning = f"Score: {score}"
                        
                        judge_scores.append(score)
                        judge_reasoning.append(reasoning)
                        
                    except Exception as e:
                        logger.warning(f"Judge {judge_name} failed on scenario {scenario.get('name')}: {e}")
                        judge_scores.append(0.0)
                        judge_reasoning.append(f"Error: {str(e)}")
                
                avg_score = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0
                judges_results[judge_name] = {
                    'score': avg_score,
                    'individual_scores': judge_scores,
                    'reasoning': f"Average across {len(judge_scores)} scenarios",
                    'individual_reasoning': judge_reasoning,
                    'rationale': judge_info.get('rationale', '')
                }
                
            except Exception as e:
                logger.warning(f"Judge {judge_name} instantiation failed: {e}")
                judges_results[judge_name] = {
                    'score': 0.0,
                    'error': str(e),
                    'rationale': judge_info.get('rationale', '')
                }
        
        return judges_results


@lru_cache(maxsize=None)
def get_metric_by_name(name: str):
    return MetricRegistry.get_by_name(name)


@lru_cache(maxsize=None)
def get_judge_by_name(name: str):
    return JudgeRegistry.get_by_name(name)

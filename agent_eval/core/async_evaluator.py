import asyncio
from concurrent.futures import ThreadPoolExecutor
from agent_eval.core.evaluator import Evaluator


class AsyncEvaluator:
    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def evaluate_async(
        self,
        prompt: str,
        model_output: str = None,
        reference_output: str = None,
        user_query: str = None,
        metrics: list = None,
        judges: list = None,
        model=None,
        prompt_optimizer: bool = False,
        max_prompt_improvements: int = 1,
        model_kwargs: dict = None,
    ):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.evaluator.evaluate(
                prompt=prompt,
                model_output=model_output,
                reference_output=reference_output,
                user_query=user_query,
                metrics=metrics,
                judges=judges,
                model=model,
                prompt_optimizer=prompt_optimizer,
                max_prompt_improvements=max_prompt_improvements,
                **(model_kwargs or {})
            ),
        )

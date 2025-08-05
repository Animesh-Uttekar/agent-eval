import hashlib
import json


class EvaluationCache:
    def __init__(self):
        self._cache = {}

    def _make_key(
        self,
        prompt: str,
        model_output: str,
        reference_output: str,
        user_query: str,
        metrics: list,
        judges: list,
        prompt_optimizer: bool,
        max_prompt_improvements: int,
    ) -> str:
        def _safe_str(x):
            return (
                str(x)
                if not isinstance(x, (str, int, float, bool, list, dict, type(None)))
                else x
            )

        key_data = json.dumps(
            {
                "prompt": _safe_str(prompt),
                "model_output": _safe_str(model_output),
                "reference_output": _safe_str(reference_output),
                "user_query": _safe_str(user_query),
                "metrics": (
                    sorted(
                        m if isinstance(m, str) else getattr(m, "criteria", str(m))
                        for m in metrics
                    )
                    if metrics
                    else []
                ),
                "judges": (
                    sorted(
                        j if isinstance(j, str) else getattr(j, "criteria", str(j))
                        for j in judges
                    )
                    if judges
                    else []
                ),
                "prompt_optimizer": prompt_optimizer,
                "max_prompt_improvements": max_prompt_improvements,
            },
            sort_keys=True,
        )
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, **kwargs):
        key = self._make_key(**kwargs)
        return self._cache.get(key)

    def set(self, result: dict, **kwargs):
        key = self._make_key(**kwargs)
        self._cache[key] = result


cache_instance = EvaluationCache()

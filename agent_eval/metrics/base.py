from abc import ABC, abstractmethod, ABCMeta


class MetricRegistry(ABCMeta):
    _registry = []
    _name_map = None
    _category_map = None

    def __init__(cls, name, bases, attrs):
        if name != "BaseMetric":
            MetricRegistry._registry.append(cls)
        super().__init__(name, bases, attrs)

    @classmethod
    def _build_maps(cls):
        cls._name_map = {}
        cls._category_map = {}
        for metric_cls in cls._registry:
            names = [metric_cls.__name__.replace("Metric", "").lower()]
            if hasattr(metric_cls, "aliases"):
                names += [alias.lower() for alias in metric_cls.aliases]
            for n in names:
                cls._name_map[n] = metric_cls

    @classmethod
    def get_by_name(cls, name):
        if cls._name_map is None:
            cls._build_maps()
        return cls._name_map.get(name.lower())

    @classmethod
    def get_by_category(cls, category):
        if cls._category_map is None:
            cls._build_maps()
        return cls._category_map.get(category, [])


class BaseMetric(ABC, metaclass=MetricRegistry):
    """
    Abstract base class for all evaluation metrics.

    This class provides the foundation for implementing various evaluation metrics
    that can assess the quality of generated text against reference text or other criteria.
    All metrics must inherit from this class and implement the evaluate method.

    Attributes:
        suggestion: String containing prompt improvement suggestions when metric score is low
    """

    suggestion = ""

    def __init__(self, criteria: str):
        self.criteria = criteria

    @abstractmethod
    def evaluate(self, generated, reference=None, **kwargs):
        """
        Evaluate a generated output against a reference or other context.

        This method must be implemented by all concrete metric classes. It should
        calculate a score that indicates how well the generated text matches or
        satisfies the evaluation criteria.

        Args:
            generated (str): The generated text to be evaluated
            reference (str, optional): The reference text to compare against
            **kwargs: Additional keyword arguments that may be needed for specific metrics

        Returns:
            dict: A dictionary containing at least a 'score' key with the evaluation result.
                  May also include 'suggestion' and 'error' keys.

        Example:
            >>> metric = BLEUMetric()
            >>> result = metric.evaluate("generated text", "reference text")
            >>> print(result['score'])
            0.75
        """
        pass

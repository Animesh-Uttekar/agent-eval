import logging
import time
import uuid
from functools import wraps
from contextvars import ContextVar

_current_task_id: ContextVar[str] = ContextVar("current_task_id", default=None)
_current_logger: ContextVar[logging.Logger] = ContextVar("current_logger", default=None)


def get_current_task_id() -> str:
    return _current_task_id.get()


def get_logger() -> logging.Logger:
    return _current_logger.get()


def _generate_task_id() -> str:
    return str(uuid.uuid4())


def _get_qualified_name(func):
    module = func.__module__
    qualname = func.__qualname__
    return f"{module}.{qualname}"


def _get_logger(task_id: str, parent_id: str = None) -> logging.Logger:
    logger = logging.getLogger(task_id)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # logger.propagate = False
        # null_handler = logging.NullHandler()
        # logger.addHandler(null_handler)

        # Dev
        import sys

        stream_handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            fmt=f"%(asctime)s | task_id={task_id} | parent_id={parent_id} | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


import inspect


def loggable(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        task_name = _get_qualified_name(func)
        parent_id = get_current_task_id()
        task_id = _generate_task_id()
        logger = _get_logger(task_id, parent_id)

        token_id = _current_task_id.set(task_id)
        token_logger = _current_logger.set(logger)

        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs)
        sanitized = {
            name: (
                "<masked>"
                # if name in ("prompt", "user_query", "model_output", "reference_output")
                if name in ()
                else val
            )
            for name, val in bound.arguments.items()
        }
        logger.info(f"\nStarted task: {task_name} \nargs={sanitized!r}\n")
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(
                f"Finished task: {task_name} \nresult={result!r} \nduration={duration:.3f}s\n"
            )
            return result
        except Exception as e:
            logger.exception(f"Error in task '{task_name}': {e}")
            raise
        finally:
            _current_task_id.reset(token_id)
            _current_logger.reset(token_logger)

    return wrapper


def enable_console_logging(level=logging.INFO):
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(handler)

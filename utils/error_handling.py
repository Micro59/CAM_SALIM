# utils/error_handling.py
"""
Centralized error handling, custom exceptions, and structured logging
for the Invisibility Cloaking System.

Features:
- Custom exception hierarchy with severity categories
- GPU OOM handling with retry / fallback / cleanup
- Execution time logging decorator
- Pipeline stage timing & structured logging
- Consistent log format to both file and console
"""

import logging
import time
import traceback
import functools
from typing import Callable, TypeVar, Optional, Any, Dict
from enum import Enum
from contextlib import contextmanager

import torch

# ────────────────────────────────────────────────
# Global logging configuration
# ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-18s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("cloaking_system.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("cloaking")


class ErrorSeverity(Enum):
    """Severity classification for errors."""
    RECOVERABLE = "recoverable"   # can continue with fallback / skip
    RETRIABLE   = "retriable"     # may succeed on retry / different device
    FATAL       = "fatal"         # stop execution


class CloakingError(Exception):
    """Base exception for all project-specific errors."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.FATAL):
        self.message = message
        self.severity = severity
        super().__init__(message)

    def __str__(self) -> str:
        return f"{self.severity.value.upper()}: {self.message}"


# Specific error types
class NoObjectsDetectedError(CloakingError):
    """No target objects (e.g. persons) found in frame."""
    def __init__(self, message: str = "No target objects detected"):
        super().__init__(message, ErrorSeverity.RECOVERABLE)


class RealismCheckFailedError(CloakingError):
    """Output realism score below acceptable threshold."""
    def __init__(self, score: float, threshold: float, message: str = None):
        msg = message or f"Realism score {score:.4f} < threshold {threshold:.4f}"
        super().__init__(msg, ErrorSeverity.RECOVERABLE)
        self.score = score
        self.threshold = threshold


class GPUMemoryError(CloakingError):
    """CUDA out-of-memory or related GPU runtime error."""
    def __init__(self, message: str = "GPU out of memory"):
        super().__init__(message, ErrorSeverity.RETRIABLE)


class ModelLoadingError(CloakingError):
    """Failed to load model weights / architecture."""
    def __init__(self, model_name: str, path: str, reason: str = None):
        msg = f"Failed to load {model_name} from {path}"
        if reason:
            msg += f" – {reason}"
        super().__init__(msg, ErrorSeverity.FATAL)


# ────────────────────────────────────────────────
# Decorators
# ────────────────────────────────────────────────
T = TypeVar('T')


def handle_gpu_errors(
    fallback: Optional[T] = None,
    retry_on_cpu: bool = False,
    clear_cache: bool = True
) -> Callable:
    """
    Decorator to gracefully handle common GPU-related exceptions.

    Args:
        fallback: Value to return if error is non-recoverable
        retry_on_cpu: Whether to automatically retry function on CPU
        clear_cache: Call torch.cuda.empty_cache() on OOM
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            func_logger = logging.getLogger(f"{func.__module__}.{func.__name__}")

            try:
                return func(*args, **kwargs)

            except torch.cuda.OutOfMemoryError as e:
                func_logger.warning(f"CUDA OOM in {func.__name__}: {e}")
                if clear_cache:
                    torch.cuda.empty_cache()
                    func_logger.debug("GPU cache cleared")

                if retry_on_cpu:
                    func_logger.info("Retrying operation on CPU...")
                    kwargs["device"] = "cpu"  # assumes function accepts device kwarg
                    return func(*args, **kwargs)

                if fallback is not None:
                    func_logger.info("Returning fallback value")
                    return fallback

                raise GPUMemoryError(str(e)) from e

            except RuntimeError as e:
                if "CUDA" in str(e) or "cu" in str(e).lower():
                    func_logger.error(f"CUDA runtime error: {e}")
                    if clear_cache:
                        torch.cuda.empty_cache()
                    raise GPUMemoryError(str(e)) from e
                raise

        return wrapper
    return decorator


def log_execution_time(logger_name: str = None):
    """Decorator that logs function execution duration."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log = logging.getLogger(logger_name or func.__module__)
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            log.debug(f"{func.__qualname__} finished in {elapsed:.3f}s")
            return result
        return wrapper
    return decorator


# ────────────────────────────────────────────────
# Structured Pipeline Logger
# ────────────────────────────────────────────────
class PipelineLogger:
    """Helper for logging pipeline stages with timing and context."""

    def __init__(self, name: str = "Pipeline"):
        self.logger = logging.getLogger(name)
        self._stage_starts: Dict[str, float] = {}

    @contextmanager
    def stage(self, name: str, extra: Dict[str, Any] = None):
        """Context manager for timing and logging a processing stage."""
        start = time.perf_counter()
        self._stage_starts[name] = start
        self.logger.info(f"→ Starting stage: {name}")
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            msg = f"← Completed {name} in {elapsed:.3f}s"
            if extra:
                msg += " | " + " ".join(f"{k}={v}" for k, v in extra.items())
            self.logger.info(msg)

    def error(self, exc: Exception, context: str = "", fatal: bool = False):
        """Log exception with full traceback."""
        level = self.logger.error if fatal else self.logger.exception
        level(f"Error during {context or 'processing'}: {exc}")
        if not isinstance(exc, CloakingError):
            self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")

    def warning(self, msg: str):
        self.logger.warning(msg)

    def info(self, msg: str):
        self.logger.info(msg)

    def result_summary(self, result: Dict[str, Any]):
        """Log final processing result summary."""
        parts = []
        if "processing_time" in result:
            parts.append(f"time={result['processing_time']:.3f}s")
        if "detections" in result:
            parts.append(f"detections={len(result['detections'])}")
        if "realism_score" in result:
            parts.append(f"realism={result['realism_score']:.4f}")
        if parts:
            self.logger.info("Result: " + " | ".join(parts))


# Global / convenience instances
pipeline_logger = PipelineLogger("CloakingPipeline")


if __name__ == "__main__":
    # Quick test / demonstration
    logger.info("Error handling utilities loaded")
    try:
        raise NoObjectsDetectedError()
    except CloakingError as e:
        pipeline_logger.error(e, context="test", fatal=False)

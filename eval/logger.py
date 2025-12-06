import logging


class _ExtraFieldsFormatter(logging.Formatter):
    """Custom formatter that includes extra fields in log output."""

    def format(self, record: logging.LogRecord) -> str:
        # Get the base formatted message
        base_message = super().format(record)

        # Extract extra fields (fields not in the default LogRecord)
        default_keys = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "thread",
            "threadName",
            "exc_info",
            "exc_text",
            "stack_info",
            "asctime",
            "taskName",
        }

        extra_fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in default_keys and value is not None
        }

        # Append extra fields to the message if they exist
        if extra_fields:
            extra_str = " | ".join(
                f"{key}={value}" for key, value in extra_fields.items()
            )
            return f"{base_message} | {extra_str}"

        return base_message


def configure_logger(name: str = "eval", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger with custom formatting.

    Args:
        name: Name for the logger instance.
        level: Logging level (default: INFO).

    Returns:
        Configured logger instance.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        _ExtraFieldsFormatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False  # Prevent duplicate logs

    return logger


def add_file_handler(logger: logging.Logger, log_path: str) -> None:
    """Add a file handler to an existing logger.

    Args:
        logger: Logger instance to configure.
        log_path: Path to the log file.
    """
    from pathlib import Path

    log_file = Path(log_path).resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        _ExtraFieldsFormatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)
    logger.info("File logging enabled", extra={"log_path": str(log_file)})

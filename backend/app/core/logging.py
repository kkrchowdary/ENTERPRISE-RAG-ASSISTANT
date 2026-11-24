import asyncio
import json
import logging
import logging.handlers
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from starlette.datastructures import MutableHeaders
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
LOG_FILE = Path(__file__).resolve().parents[2] / "logs" / "app.log"


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id.get()
        return True


class JsonFormatter(logging.Formatter):
    fields = (
        "request_id", "method", "route", "status_code", "duration_ms",
        "completed", "error_type", "chunk_count", "candidate_count",
        "semantic_count", "lexical_count", "source_count", "size_bytes",
        "tool", "reason",
    )

    def format(self, record: logging.LogRecord) -> str:
        data = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        data.update({name: getattr(record, name) for name in self.fields if hasattr(record, name)})
        return json.dumps(data, ensure_ascii=False)


class RagLogHandler(logging.StreamHandler):
    pass


class RagLogFileHandler(logging.handlers.RotatingFileHandler):
    pass


class SafeServerErrorFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info:
            error_type = record.exc_info[0].__name__
            record.msg = "Server exception (%s); see application logs"
            record.args = (error_type,)
            record.exc_info = None
            record.exc_text = None
        return True


def configure_logging() -> None:
    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)
    if not any(isinstance(handler, RagLogHandler) for handler in logger.handlers):
        handler = RagLogHandler(sys.stderr)
        handler.addFilter(RequestContextFilter())
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    file_handler = next(
        (handler for handler in logger.handlers if isinstance(handler, RagLogFileHandler)),
        None,
    )
    if file_handler is None:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RagLogFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.addFilter(RequestContextFilter())
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)
    logger.propagate = False
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    server_logger = logging.getLogger("uvicorn.error")
    if file_handler not in server_logger.handlers:
        server_logger.addHandler(file_handler)
    if not any(isinstance(item, SafeServerErrorFilter) for item in server_logger.filters):
        server_logger.addFilter(SafeServerErrorFilter())


@contextmanager
def log_operation(logger: logging.Logger, name: str):
    start = time.perf_counter()
    logger.info("%s.started", name)
    try:
        yield
    except asyncio.CancelledError:
        logger.info("%s.cancelled", name, extra={
            "duration_ms": round((time.perf_counter() - start) * 1000, 2)
        })
        raise
    except Exception as exc:
        logger.error("%s.failed", name, extra={
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            "error_type": type(exc).__name__,
        })
        raise
    else:
        logger.info("%s.completed", name, extra={
            "duration_ms": round((time.perf_counter() - start) * 1000, 2)
        })


class RequestLoggingMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app
        self.logger = logging.getLogger("app.requests")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        current_request_id = uuid4().hex
        token = request_id.set(current_request_id)
        start = time.perf_counter()
        status_code = None
        completed = False
        method = scope.get("method", "")
        method = method if method in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"} else "OTHER"
        self.logger.info("request.started", extra={"method": method})

        async def send_with_context(message: Message) -> None:
            nonlocal status_code, completed
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message)["X-Request-ID"] = current_request_id
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                completed = True

        try:
            await self.app(scope, receive, send_with_context)
        except asyncio.CancelledError:
            self.logger.info("request.cancelled")
            raise
        except Exception as exc:
            self.logger.error("request.failed", extra={"error_type": type(exc).__name__})
            if status_code is None:
                await PlainTextResponse("Internal Server Error", status_code=500)(
                    scope, receive, send_with_context
                )
            raise
        finally:
            route = getattr(scope.get("route"), "path", "unmatched")
            self.logger.info("request.completed", extra={
                "method": method,
                "route": route,
                "status_code": status_code,
                "completed": completed,
                "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            })
            request_id.reset(token)
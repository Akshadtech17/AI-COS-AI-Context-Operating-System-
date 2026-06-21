"""
Observability setup: OpenTelemetry tracing + Sentry error tracking.

Both integrations are optional and enabled only when their configuration is present:
  AICOS_OTEL_ENDPOINT=http://jaeger:4317   → OpenTelemetry OTLP gRPC export
  AICOS_SENTRY_DSN=https://...@sentry.io/… → Sentry error & performance tracking

When the required libraries are not installed, configure_telemetry() logs a warning
and skips that integration — the application works normally without it.

OpenTelemetry auto-instrumentation (when enabled):
  - FastAPI: traces every HTTP request (method, path, status, latency)
  - SQLAlchemy: traces every database query

Sentry auto-instrumentation (when enabled):
  - FastAPI: captures unhandled exceptions and slow transactions
  - SQLAlchemy: captures slow queries as breadcrumbs
"""

from __future__ import annotations

from aicos.core.logging import get_logger

log = get_logger("core.telemetry")


def configure_telemetry(
    otel_endpoint: str | None,
    sentry_dsn: str | None,
    service_name: str = "aicos-gateway",
    environment: str = "production",
) -> None:
    """
    Configure OTel and/or Sentry based on the provided settings.
    Call once at application startup before routes are registered.
    """
    if otel_endpoint:
        _setup_otel(otel_endpoint, service_name, environment)

    if sentry_dsn:
        _setup_sentry(sentry_dsn, environment)


def _setup_otel(endpoint: str, service_name: str, environment: str) -> None:
    """Configure OpenTelemetry SDK with OTLP gRPC export."""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        log.warning(
            "opentelemetry-sdk not installed — tracing disabled. Fix: pip install 'aicos[otel]'"
        )
        return

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    except ImportError:
        log.warning(
            "opentelemetry-exporter-otlp not installed — tracing disabled. "
            "Fix: pip install 'aicos[otel]'"
        )
        return

    resource = Resource.create(
        {
            "service.name": service_name,
            "deployment.environment": environment,
        }
    )
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    _instrument_fastapi()
    _instrument_sqlalchemy()

    log.info("OpenTelemetry tracing enabled", extra={"endpoint": endpoint})


def _instrument_fastapi() -> None:
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor().instrument()
    except ImportError:
        log.warning("opentelemetry-instrumentation-fastapi not installed — FastAPI spans skipped")


def _instrument_sqlalchemy() -> None:
    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        SQLAlchemyInstrumentor().instrument()
    except ImportError:
        log.warning("opentelemetry-instrumentation-sqlalchemy not installed — DB spans skipped")


def _setup_sentry(dsn: str, environment: str) -> None:
    """Initialise Sentry SDK with FastAPI + SQLAlchemy integrations."""
    try:
        import sentry_sdk
    except ImportError:
        log.warning(
            "sentry-sdk not installed — error tracking disabled. Fix: pip install 'aicos[sentry]'"
        )
        return

    integrations: list[object] = []

    try:
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration

        integrations.extend([StarletteIntegration(), FastApiIntegration()])
    except ImportError:
        pass

    try:
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

        integrations.append(SqlalchemyIntegration())
    except ImportError:
        pass

    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        integrations=integrations,
        traces_sample_rate=0.1,
        profiles_sample_rate=0.05,
        send_default_pii=False,
    )

    log.info("Sentry error tracking enabled")

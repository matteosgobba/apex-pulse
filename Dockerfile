FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    F1_PREDICTION_PROJECT_ROOT=/app

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir .

COPY configs ./configs
COPY deploy/entrypoint.sh /usr/local/bin/apex-pulse-entrypoint

RUN useradd --create-home --uid 10001 apex-pulse \
    && mkdir -p /runtime \
    && chown -R apex-pulse:apex-pulse /app /runtime \
    && chmod 0755 /usr/local/bin/apex-pulse-entrypoint

USER apex-pulse

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/apex-pulse-entrypoint"]
CMD ["python", "-m", "f1_prediction.cli", "dashboard-api"]

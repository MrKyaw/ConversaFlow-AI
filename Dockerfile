FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONVERSAFLOW_STORE=sqlite \
    CONVERSAFLOW_MODEL_PATH=/app/models/nlu.json

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir . \
    && python -m conversaflow train --data-dir /app/data --output /app/models/nlu.json \
    && groupadd --gid 10001 conversaflow \
    && useradd --uid 10001 --gid 10001 --no-create-home conversaflow \
    && mkdir -p /app/var \
    && chown -R conversaflow:conversaflow /app/var
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["conversaflow", "serve", "--host", "0.0.0.0", "--port", "8000"]

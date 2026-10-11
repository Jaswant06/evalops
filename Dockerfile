# One image for the EvalOps API and the DocQA service (different commands in
# docker-compose.yml). Ollama stays on the host, where it can use the GPU.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    OLLAMA_HOST=http://host.docker.internal:11434

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[postgres,mlflow]"

# Defaults baked in; docker-compose mounts the live versions over them.
COPY configs ./configs
COPY prompts ./prompts
COPY datasets ./datasets

EXPOSE 8000 8001
CMD ["evalops", "serve", "--host", "0.0.0.0", "--port", "8000"]

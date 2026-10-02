# The live question-answering service (raglab.server). Needs RAGLAB_TOKEN, and
# RAGLAB_API_KEY, RAGLAB_MODEL and RAGLAB_EMBED_MODEL to answer with real models.
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY raglab ./raglab
RUN pip install --no-cache-dir ".[server]"
COPY corpus ./corpus
ENV PORT=8080 PYTHONUNBUFFERED=1
CMD ["sh", "-c", "exec uvicorn raglab.server:app --factory --host 0.0.0.0 --port ${PORT}"]

FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
ENV PYTHONUNBUFFERED=1 UV_PYTHON_DOWNLOADS=never ECO_DATA_DIR=/data
WORKDIR /app
RUN pip install --no-cache-dir uv==0.13.5
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY ecoportal ecoportal
COPY forked forked
COPY examples examples
COPY data data
RUN mkdir -p /data && chown 10001:10001 /data
USER 10001:10001
EXPOSE 7860
CMD [".venv/bin/uvicorn", "ecoportal.app:app", "--host", "0.0.0.0", "--port", "7860"]

# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /app

COPY requirements.txt .

RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Stage 2: Final image
FROM python:3.11-slim as base

WORKDIR /app

COPY --from=builder /wheels /wheels

RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl

COPY . .

EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000", "--workers", "4"]

FROM python:3.11-slim

WORKDIR /app

COPY requirements-docker.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements-docker.txt

COPY . .

COPY src/serving/model /app/model

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

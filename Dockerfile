FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src

# Модель обучается при сборке: образ сразу готов отвечать на запросы
RUN python -m churn_monitoring.train

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "churn_monitoring.api:app", "--host", "0.0.0.0", "--port", "8000"]

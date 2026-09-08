FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn
COPY recsys_loom recsys_loom
COPY services services
COPY scripts scripts
EXPOSE 8000
CMD ["uvicorn", "services.recommender.app:app", "--host", "0.0.0.0", "--port", "8000"]

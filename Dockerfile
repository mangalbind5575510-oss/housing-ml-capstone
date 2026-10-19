FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ ./common/
COPY training/ ./training/
COPY app/ ./app/

RUN python training/generate_dataset.py && python training/train_pipeline.py

EXPOSE 5000

CMD gunicorn --bind 0.0.0.0:${PORT:-5000} --chdir /app app.main:app

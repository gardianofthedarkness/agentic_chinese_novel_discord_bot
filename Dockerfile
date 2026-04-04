FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source
COPY app/      ./app/
COPY infra/    ./infra/
COPY helpers/  ./helpers/
COPY utils/    ./utils/
COPY db/       ./db/
COPY config.py models.py ./

ENV PYTHONIOENCODING=utf-8
ENV PYTHONUNBUFFERED=1

EXPOSE 5005

CMD ["python", "-m", "uvicorn", "utils.astream:app", "--host", "0.0.0.0", "--port", "5005"]

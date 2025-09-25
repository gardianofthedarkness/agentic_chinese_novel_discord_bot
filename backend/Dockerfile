# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for PostgreSQL
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies (including PostgreSQL support)
RUN pip install --no-cache-dir psycopg2-binary && \
    echo "Starting individual package installation..." && \
    echo "Requirements file contents:" && \
    cat requirements.txt && \
    total=$(wc -l < requirements.txt | tr -d ' ') && \
    count=0 && \
    while read -r pkg; do \
        count=$((count+1)); \
        echo "===== [INSTALLING $count/$total]: $pkg ====="; \
        pip install --no-cache-dir -vvv "$pkg" || echo "Error installing $pkg"; \
        echo "===== [COMPLETED $count/$total]: $pkg ====="; \
    done < requirements.txt && \
    echo "Successfully installed $count packages"

# Copy the current directory contents into the container
COPY . .

# Set environment variables for unified processing
ENV PYTHONIOENCODING=utf-8
ENV PYTHONPATH=/app

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches (can be overridden)
CMD ["python", "app.py"]

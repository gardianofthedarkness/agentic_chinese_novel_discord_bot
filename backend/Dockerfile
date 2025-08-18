# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install each package individually with maximum verbosity
RUN echo "Starting individual package installation..." && \
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

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]

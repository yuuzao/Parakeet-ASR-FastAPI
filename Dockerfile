FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better caching
COPY app/requirements.txt /app/

# using pip mirror
RUN pip config set global.index-url https://mirrors.huaweicloud.com/repository/pypi/simple

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app/ /app/

# Create static directory if it doesn't exist
RUN mkdir -p /app/static

# Copy the index.html file to static directory
RUN if [ -f /app/index.html ]; then \
    cp /app/index.html /app/static/; \
    fi

# Expose the port defined in the .env file (default: 8777)
EXPOSE 8777

# Start the application
CMD ["python", "main.py"]

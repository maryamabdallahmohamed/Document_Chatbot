# Multi-platform support
FROM python:3.12.0

# Set working directory
WORKDIR /app
# Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .

# Install Python dependencies with platform-specific optimizations
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories

RUN mkdir -p \
    cache \
    config \
    data \
    embedder_model_cache \
    logs \
    src \
    vectorstore
# Switch to non-root user
USER appuser
# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 5050

# Default command
CMD ["python", "api_server.py"]
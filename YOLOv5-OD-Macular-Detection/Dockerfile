# Stage 1: Builder stage to install Python, pip, and dependencies
FROM python:3.10-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build dependencies using apt-get since python:3.10-slim is Debian-based
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    bash \
    ffmpeg \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libfreetype6-dev \
    libopenblas-dev \
    && pip install --upgrade pip setuptools wheel && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements files
COPY ./yolov5/requirements.txt /app/yolov5/

# Install Python dependencies
RUN pip install --no-cache-dir -r ./yolov5/requirements.txt

# Install additional Python packages
RUN pip install --no-cache-dir scikit-image blinker python-multipart

# Stage 2: Final runtime stage using nvidia/cuda runtime (Ubuntu 22.04)
FROM nvidia/cuda:12.0.1-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create the /app directory
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl bash wget ffmpeg libsm6 libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy Python and installed dependencies from builder stage
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

# Copy the application files
COPY ./app.py /app
COPY ./yolov5 /app/yolov5
COPY ./configs /app/configs
COPY ./__init__.py /app/
# Expose the application port
EXPOSE 7003

# Run the application with uvicorn
CMD ["python","app.py"]

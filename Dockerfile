# STAGE 1: Builder (Use 3.11 to match your dev environment)
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app
COPY requirements.txt .

# Install with --prefix
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Download NLTK data to a specific folder in the builder
RUN python -m pip install nltk && \
    python -m nltk.downloader -d /install/nltk_data stopwords punkt

# STAGE 2: Runner
FROM python:3.11-slim-bookworm

WORKDIR /app

# Copy the packages (Note: folder path updated to 3.11)
COPY --from=builder /install/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /install/bin /usr/local/bin
# Copy the NLTK data
COPY --from=builder /install/nltk_data /usr/share/nltk_data

# Set Environment Variables for NLTK and CPU-only Torch
ENV NLTK_DATA=/usr/share/nltk_data
ENV CUDA_VISIBLE_DEVICES=-1

# Copy your app code
COPY app.py .
COPY model ./model
COPY src ./src

# Clean up
RUN find /usr/local -name "*.pyc" -delete && \
    find /usr/local -name "__pycache__" -delete

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
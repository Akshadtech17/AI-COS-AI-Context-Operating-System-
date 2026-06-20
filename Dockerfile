FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[all]"

# Copy source
COPY . .

# Create data directory
RUN mkdir -p /data

ENV AICOS_DB_PATH=/data/aicos.db
ENV AICOS_GATEWAY_HOST=0.0.0.0
ENV AICOS_GATEWAY_PORT=4000

EXPOSE 4000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:4000/health || exit 1

CMD ["aicos", "start"]

FROM python:3.11-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project definition & code
COPY pyproject.toml README.md ./
COPY app ./app

# Install python packages
RUN pip install --no-cache-dir -e . greenlet asyncpg psycopg2-binary

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

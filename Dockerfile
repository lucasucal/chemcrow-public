FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY setup.py README.md ./
COPY chemcrow ./chemcrow
RUN pip install --no-cache-dir -e . requests psutil

COPY run_agent.py query_agent.py ./
ENTRYPOINT ["python", "run_agent.py"]

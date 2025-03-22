FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required for pyaudio and other packages
RUN apt-get update && apt-get install -y \
    gcc \
    portaudio19-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for package management
RUN pip install uv

# Copy project files
COPY . .

# Install dependencies using uv with --system flag since we don't need a venv in Docker
RUN uv pip install --system .

EXPOSE 8501

# Add PYTHONPATH to recognize src directory
ENV PYTHONPATH=/app

CMD ["streamlit", "run", "src/app.py"]
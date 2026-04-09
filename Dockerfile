FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/

# Install Python dependencies
RUN pip install --no-cache-dir .

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run the API server
CMD ["uvicorn", "physicalfish.api.app:create_app", "--host", "0.0.0.0", "--port", "8000", "--factory"]

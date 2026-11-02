FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.13.4
COPY pyproject.toml uv.lock .python-version ./
# Include test dependencies so the video can show pytest inside this same image.
RUN uv sync --frozen
COPY . .
RUN useradd --uid 10001 --create-home appuser \
    && mkdir -p /app/.cache /app/artifacts \
    && chown -R 10001:10001 /app/.cache /app/artifacts
USER 10001:10001
EXPOSE 8501
CMD ["uv", "run", "--frozen", "streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]

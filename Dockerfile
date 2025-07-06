FROM python:3.10-slim AS builder

WORKDIR /app

COPY requirements.txt .

RUN pip install --user --no-cache-dir -r requirements.txt


FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local

ENV PATH=/root/.local/bin:$PATH

COPY . .

EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost:8501/ || exit 1
  
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501"]

FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY proxy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy proxy code
COPY proxy/main.py .

# Copy web UI as static files
COPY web/ /app/static/

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

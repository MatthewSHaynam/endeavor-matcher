FROM python:3.8-slim

# Install build dependencies (if needed for compiling numpy)
RUN apt-get update && apt-get install -y gcc build-essential

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]

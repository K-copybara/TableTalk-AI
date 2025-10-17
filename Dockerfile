# 1. Base image
FROM python:3.13-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 시스템 의존성 설치 (중요)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    librdkafka-dev \
 && rm -rf /var/lib/apt/lists/*

# 4. requirements 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 앱 복사
COPY . .

# 6. FastAPI 실행 (uvicorn)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

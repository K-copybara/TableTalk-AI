# 1. Base image
FROM python:3.13-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# 3. requirements 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 앱 복사
COPY . .

# 5. FastAPI 실행 (uvicorn)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

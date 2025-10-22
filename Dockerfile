# 1. Python 3.12 버전의 깨끗한 OS로 시작합니다.
FROM python:3.12-slim

# 2. (핵심!) Java(JDK) 17 버전을 먼저 설치합니다.
# openjdk-17-jdk 대신, 이 시스템의 기본 Java 패키지(default-jdk)를 설치합니다.
RUN apt-get update && apt-get install -y default-jdk && rm -rf /var/lib/apt/lists/*

# 3. /app 이라는 작업 폴더를 만듭니다.
WORKDIR /app

# 4. requirements.txt 파일을 먼저 복사해서 라이브러리를 설치합니다.
# (이렇게 하면 나중에 코드만 바뀔 때 빌드 속도가 빨라집니다.)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 5. 나머지 모든 프로젝트 파일(.py, .pkl 등)을 /app 폴더로 복사합니다.
# (Render는 Git LFS를 지원하므로 .pkl 파일도 잘 복사됩니다.)
COPY . .

# 6. gunicorn 서버를 실행합니다. 
# (Render는 10000번 포트를 기본으로 사용합니다.)
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "api_server:app"]
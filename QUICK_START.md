# SolarX 빠른 시작 가이드 (Quick Start Guide)

이 가이드는 SolarX 웹 애플리케이션을 로컬에서 테스트하고 Docker로 배포하는 방법을 설명합니다.

This guide explains how to test the SolarX web application locally and deploy it with Docker.

## 📋 목차 (Table of Contents)

1. [백엔드만 테스트 (Backend Only)](#1-백엔드만-테스트-backend-only)
2. [프론트엔드만 테스트 (Frontend Only)](#2-프론트엔드만-테스트-frontend-only)
3. [전체 스택 로컬 개발 (Full Stack Local)](#3-전체-스택-로컬-개발-full-stack-local)
4. [Docker로 실행 (Run with Docker)](#4-docker로-실행-run-with-docker)
5. [문제 해결 (Troubleshooting)](#5-문제-해결-troubleshooting)

---

## 1. 백엔드만 테스트 (Backend Only)

### 설치 (Installation)

```bash
# 1. backend 디렉토리로 이동
cd backend

# 2. Python 패키지 설치 (Python 3.9-3.12 필요)
pip install -r requirements.txt
```

### 테스트 실행 (Run Tests)

```bash
# 프로젝트 루트로 돌아가기
cd ..

# 테스트 스크립트 실행
python test_backend.py
```

**예상 출력:**
```
============================================================
SolarX Backend Test Suite
============================================================
Testing backend imports...
  [OK] FastAPI app imported successfully

Testing API routes...
  [OK] /api/health
  [OK] /api/vendors
  [OK] /api/simulate/benchmark
  ...

Total: 3/3 tests passed

>>> Backend is ready!
```

### 백엔드 서버 실행 (Start Backend Server)

**방법 1: 실행 스크립트 사용 (권장):**
```bash
cd backend
python run_server.py

# 또는 Windows에서
start_server.bat
```

**방법 2: Uvicorn 직접 실행:**
```bash
# 프로젝트 루트에서
uvicorn backend.app.main:app --reload --port 8000

# 또는 backend 디렉토리에서
cd backend
uvicorn app.main:app --reload --port 8000
```

**접속 URL:**
- API 문서: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/health
- Vendors API: http://localhost:8000/api/vendors

**테스트 명령어:**
```bash
# Health check
curl http://localhost:8000/api/health

# Get vendors
curl http://localhost:8000/api/vendors

# Run benchmark simulation (takes ~30 seconds)
curl -X POST http://localhost:8000/api/simulate/benchmark \
  -H "Content-Type: application/json" \
  -d '{"battery_capacity_kwh": 2280, "charge_threshold": 0.9, "discharge_threshold": 1.1, "allow_grid_charge": true, "region_factor": 1.0}'
```

---

## 2. 프론트엔드만 테스트 (Frontend Only)

### 설치 (Installation)

```bash
# 1. frontend 디렉토리로 이동
cd frontend

# 2. Node.js 패키지 설치 (Node.js 18+ 필요)
npm install
```

### 개발 서버 실행 (Start Dev Server)

```bash
npm run dev
```

**접속 URL:** http://localhost:5173

**주의:** 프론트엔드만 실행하면 API 호출이 실패합니다. 백엔드도 함께 실행해야 합니다.

**Note:** If you run only the frontend, API calls will fail. You need to run the backend as well.

### 프로덕션 빌드 (Production Build)

```bash
# 프론트엔드 빌드
npm run build

# 빌드 결과물은 frontend/dist/에 생성됩니다
ls dist/
```

---

## 3. 전체 스택 로컬 개발 (Full Stack Local)

백엔드와 프론트엔드를 동시에 실행합니다.

Run both backend and frontend simultaneously.

### Terminal 1: 백엔드

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### Terminal 2: 프론트엔드

```bash
cd frontend
npm run dev
```

### 접속

- **프론트엔드 개발 서버:** http://localhost:5173
- **백엔드 API:** http://localhost:8000
- **API 문서:** http://localhost:8000/docs

프론트엔드는 Vite proxy를 통해 자동으로 백엔드 API (localhost:8000)에 연결됩니다.

The frontend automatically connects to the backend API (localhost:8000) via Vite proxy.

---

## 4. Docker로 실행 (Run with Docker)

Docker를 사용하면 백엔드와 프론트엔드가 하나의 컨테이너에서 실행됩니다.

With Docker, both backend and frontend run in a single container.

### 방법 1: Docker Compose (권장)

```bash
# 프로젝트 루트에서 실행
docker-compose up

# 중지
docker-compose down
```

### 방법 2: Docker CLI

```bash
# 1. Docker 이미지 빌드 (5-10분 소요)
docker build -t solarx-web .

# 2. 컨테이너 실행
docker run -p 8000:8000 solarx-web

# 3. 접속
# http://localhost:8000
```

### Docker로 실행 시 접속 URL

- **웹 UI:** http://localhost:8000
- **API 문서:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/api/health

**주의:** Docker 빌드 시 프론트엔드가 자동으로 빌드됩니다.

**Note:** The frontend is automatically built during Docker build.

---

## 5. 문제 해결 (Troubleshooting)

### 문제: `ModuleNotFoundError: No module named 'fastapi'`

**해결방법:**
```bash
cd backend
pip install -r requirements.txt
```

### 문제: `uvicorn app.main:app` 실행 시 오류

**증상:** backend 디렉토리에서 uvicorn 명령어 실행 시 모듈을 찾지 못함

**해결방법 1 (권장):**
```bash
cd backend
python run_server.py
```

**해결방법 2:**
프로젝트 루트에서 실행
```bash
cd C:\dev\SolarX
uvicorn backend.app.main:app --reload --port 8000
```

### 문제: `torch==2.1.0` 설치 실패 (Python 3.12)

**해결방법:** requirements.txt가 이미 수정되었습니다. 최신 버전 사용:
```
torch>=2.2.0
```

재설치:
```bash
cd backend
pip install -r requirements.txt --upgrade
```

### 문제: 프론트엔드에서 API 호출 실패

**확인사항:**
1. 백엔드가 실행 중인지 확인: http://localhost:8000/api/health
2. CORS 설정 확인 (`backend/app/config.py`)
3. 프론트엔드 proxy 설정 확인 (`frontend/vite.config.ts`)

### 문제: Docker 빌드 실패

**확인사항:**
1. Docker가 설치되어 있는지 확인: `docker --version`
2. Docker Desktop이 실행 중인지 확인
3. `.dockerignore` 파일이 있는지 확인

**빌드 로그 확인:**
```bash
docker build -t solarx-web . --progress=plain
```

### 문제: Docker 컨테이너가 즉시 종료됨

**로그 확인:**
```bash
docker logs <container-id>
```

**일반적인 원인:**
- LSTM 모델 파일 누락: `SolarX/src/lstm_solar_model.pth`
- 데이터 파일 누락: `SolarX/data/`

---

## 📚 추가 자료 (Additional Resources)

- **Docker 배포 가이드:** `DOCKER_DEPLOYMENT.md`
- **CLAUDE.md:** 프로젝트 전체 문서
- **API 문서:** http://localhost:8000/docs (서버 실행 후)

---

## ✅ 체크리스트 (Checklist)

개발 환경 설정:
- [ ] Python 3.9-3.12 설치됨
- [ ] Node.js 18+ 설치됨 (프론트엔드용)
- [ ] Docker 설치됨 (배포용)
- [ ] backend requirements.txt 설치됨
- [ ] LSTM 모델 파일 존재: `SolarX/src/lstm_solar_model.pth`
- [ ] 데이터 파일 존재: `SolarX/data/`

테스트:
- [ ] `python test_backend.py` 통과
- [ ] 백엔드 서버 실행 성공: `uvicorn app.main:app --reload`
- [ ] 프론트엔드 서버 실행 성공: `npm run dev`
- [ ] Docker 빌드 성공: `docker build -t solarx-web .`
- [ ] Docker 컨테이너 실행 성공: `docker run -p 8000:8000 solarx-web`

---

## 🚀 다음 단계 (Next Steps)

1. **로컬 개발 완료 후:**
   - GitHub에 푸시
   - GitHub Actions가 자동으로 Docker 이미지 빌드
   - GHCR (GitHub Container Registry)에 푸시

2. **클라우드 배포:**
   - Google Cloud Run (권장)
   - Fly.io
   - Heroku
   - 자세한 내용: `DOCKER_DEPLOYMENT.md`

---

**마지막 업데이트:** 2026-02-15
**작성자:** Claude Code

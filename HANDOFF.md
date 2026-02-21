# HANDOFF - SolarX 프로젝트 인계 문서

**최종 업데이트:** 2026-02-21
**작성자:** Claude Code (Sonnet 4.6) + Codex (GPT-5) + Claude Code (Opus 4.6)
**목적:** 다음 에이전트가 이 파일만 읽고 작업을 이어갈 수 있도록 현재 상태 정리

> ⚠️ **중요:** 이 문서는 프로젝트의 현재 상태를 정확히 반영합니다.

---

## 🎯 한눈에 보는 프로젝트 상태

| 항목 | 상태 | 비고 |
|------|------|------|
| **코드 작성** | ✅ 완료 | Backend (FastAPI) + Frontend (React) |
| **Docker 설정** | ✅ 완료 | Dockerfile, docker-compose.yml |
| **GitHub Actions** | ✅ 완료 | CI/CD → GHCR 자동 빌드/푸시 |
| **UI/UX 개선** | ✅ 완료 | 2026-02-20 대규모 개선 |
| **빌드** | ✅ 통과 | TypeScript 오류 없음, vite build 성공 |
| **배포** | ✅ 완료 | Render에 배포됨 |

**→ 라이브 URL:** https://solarx-5pbh.onrender.com

---

## 🔄 최근 세션 업데이트 (2026-02-21)

이번 세션은 **배포 플랫폼 선정 및 Render 배포** 작업이었다.

### 시도한 것
- 배포 플랫폼 비교 분석: Render, Railway, Google Cloud Run, Fly.io, Vercel
- Vercel은 Docker 미지원 + 프론트엔드 전용이라 SolarX(풀스택 Docker)에 부적합 → 제외
- Render를 최종 선택 (가장 쉬움, Docker 자동 감지, GitHub 연동 자동 배포)

### 성공한 것
- ✅ Render 배포 완료 → https://solarx-5pbh.onrender.com
- ✅ Favicon 추가 (apple-touch-icon, 16x16, 32x32, 512x512, SVG)
- ✅ 한국어 번역 수정 및 Landing 페이지 개선 커밋

### 실패한 것 / 미완료
- ❌ CORS 프로덕션 설정 미적용: `backend/app/main.py:49`에서 여전히 `allow_origins=["*"]`
  - 같은 도메인 서빙이라 현재 문제없지만, 보안을 위해 변경 권장
  - `backend/app/config.py`의 `cors_origins` → 환경변수 `CORS_ORIGINS`로 오버라이드 가능 (pydantic-settings)
  - `main.py`에서 `["*"]` → `settings.cors_origins` 변경 필요

---

## 🔄 이전 세션 업데이트 (2026-02-20)

이번 세션은 **프론트엔드 UI/UX 개선** 작업이었다. 기능 변경 없음, 배포 관련 변경 없음.

### 0. ✅ 추가 업데이트 (2026-02-20, Codex 반영)

- 백엔드 실행 설정 외부화: `backend/app/config.py`에 `host`, `port` 설정 추가. `backend/run_server.py`에 `--host`, `--port` CLI 인자 추가.
- SOC 안내 UI 공통 컴포넌트화: `frontend/src/components/common/SOCLowInfoTooltip.tsx` 신규 추가.
- SOH 게이지 상태 기준 변경: ACC II 워런티 하한 기준 (`>=75` Healthy, `70-74.9` Warning, `<70` Critical).
- Solar 충전 애니메이션 리팩터링: Framer Motion → GSAP timeline 기반 3단계 루프 (charging→converting→selling).
- 프론트엔드 의존성 추가: `gsap` 추가.
- 랜딩 카피 미세 조정: `SolarX: AI-Powered` → `SolarX: AI Powered`.

### 1~5. UI/UX 개선 요약
- ✅ BatteryGauge SOC 툴팁 단일화 (영역당 하나)
- ✅ 차트 Y축 레이블 회전 제거 → 가로 배치, Y축 숫자 잘림 해결
- ✅ 차트 X축 단위 수정 (ProfitChart: days, PredictionChart: hours)
- ✅ CHARGING/CONVERTING/SELLING 텍스트 크게, 밝게
- ✅ 홈 타이틀 줄바꿈 ("SolarX: AI 기반" / "배터리 최적화") — `whiteSpace="nowrap"` + 키 분리
- ✅ 홈 부제목 줄바꿈 및 브랜드명 영문 통일

---

## 📍 배포 아키텍처

```
GitHub (main push)
  ├── GitHub Actions → GHCR (ghcr.io/iimmuunnee/solarx:latest)
  └── Render (자동 감지) → https://solarx-5pbh.onrender.com
         └── Docker 빌드 (multi-stage)
               ├── Stage 1: Node 18 → React/Vite 프론트엔드 빌드
               └── Stage 2: Python 3.9 → FastAPI + 프론트엔드 서빙
                     ├── / → frontend/dist (SPA, html=True)
                     └── /api/* → FastAPI 엔드포인트
```

**Render 무료 티어 특성:**
- 15분 비활성 시 슬립 → 첫 접속에 30~60초 대기
- 512MB RAM (precomputed 결과 사용 시 충분)
- GitHub push 시 자동 재배포

---

## 🚀 다음 에이전트가 해야 할 작업

### 우선순위 1 🔴 — CORS 프로덕션 설정
- `backend/app/main.py:49`의 `allow_origins=["*"]` → `allow_origins=settings.cors_origins`
- `backend/app/config.py`의 `cors_origins`에 `"https://solarx-5pbh.onrender.com"` 추가

### 우선순위 2 🟡 — 추가 UI 개선 (선택)
- 모바일 반응형 검토 (홈 타이틀 `nowrap` → 좁은 화면에서 넘칠 수 있음)
- Demo 페이지 결과 테이블 미구현 (현재 MetricsGrid만 있음)
- Architecture 페이지 실제 다이어그램 구현

### 우선순위 3 🟢 — 프로덕션 고도화 (선택)
- 환경변수 `.env` 분리
- Render 유료 플랜 전환 (슬립 방지, 메모리 증가)
- 커스텀 도메인 연결

---

## 💡 개발 환경 실행 방법

```bash
# 로컬 개발 (hot reload)
# Terminal 1
cd backend && uvicorn app.main:app --reload

# Terminal 2
cd frontend && npm run dev
# Frontend: http://localhost:5173, Backend: http://localhost:8000

# Docker (프로덕션과 동일)
docker-compose up
# http://localhost:8000
```

---

## 🐛 알려진 이슈 및 주의사항

1. **CORS** — `allow_origins=["*"]` 상태. 같은 도메인 서빙이라 당장 문제없지만 보안 개선 필요.
2. **홈 타이틀 모바일** — `whiteSpace="nowrap"` 적용된 "SolarX: AI 기반"이 아주 좁은 화면에서 넘칠 수 있음. `base: '5xl'`로 대응 중.
3. **BatteryGauge SOC 툴팁 threshold** — `SOC_LOW_THRESHOLD = 50`. `BatteryGauge.tsx`에서 export.
4. **ProfitChart Brush** — day 단위 표시. 툴팁 `"{{day}}일차"`.
5. **JSON 수정 주의** — locale JSON 수정 시 `json.dumps()` 사용 필수. Write 도구의 `\n`이 실제 newline으로 삽입되어 JSON 파싱 오류 발생.
6. **Render 슬립** — 무료 티어는 15분 비활성 시 슬립. 첫 접속 30~60초 대기.

---

## ✅ 전체 완료된 작업 이력

### 2026-02-21
- ✅ 배포 플랫폼 분석 및 Render 선정
- ✅ Render 배포 완료 (https://solarx-5pbh.onrender.com)
- ✅ Favicon 에셋 추가 (5종)
- ✅ Landing 페이지/번역 미세 조정 커밋

### 2026-02-20
- ✅ BatteryGauge 툴팁 단일화, 차트 축 개선, 애니메이션 개선
- ✅ 홈 타이틀 줄바꿈, 부제목 브랜드명 통일

### 2026-02-17
- ✅ GitHub Actions, Docker 컨테이너화, Backend/Frontend 구현, CI/CD, 문서화

---

**마지막 업데이트:** 2026-02-21
**라이브 URL:** https://solarx-5pbh.onrender.com
**빌드 상태:** ✅ `npm run build` 통과 (오류 없음)
**배포 상태:** ✅ Render 배포 완료

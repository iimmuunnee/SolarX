# HANDOFF - SolarX 프로젝트 인계 문서

**최종 업데이트:** 2026-02-20
**작성자:** Claude Code (Sonnet 4.6) + Codex (GPT-5)
**목적:** 다음 에이전트가 이 파일만 읽고 작업을 이어갈 수 있도록 현재 상태 정리

> ⚠️ **중요:** 이 문서는 프로젝트의 현재 상태를 정확히 반영합니다.

---

## 🎯 한눈에 보는 프로젝트 상태

| 항목 | 상태 | 비고 |
|------|------|------|
| **코드 작성** | ✅ 완료 | Backend (FastAPI) + Frontend (React) |
| **Docker 설정** | ✅ 완료 | Dockerfile, docker-compose.yml |
| **GitHub Actions** | ✅ 완료 | CI/CD 정상 작동 |
| **UI/UX 개선** | ✅ 완료 | 2026-02-20 대규모 개선 |
| **빌드** | ✅ 통과 | TypeScript 오류 없음, vite build 성공 |
| **배포** | ⚠️ 미완료 | 클라우드 배포 대기 중 |

**→ 다음 단계:** 클라우드 배포 (Google Cloud Run / Fly.io / Heroku)

---

## 🔄 최근 세션 업데이트 (2026-02-20)

이번 세션은 **프론트엔드 UI/UX 개선** 작업이었다. 기능 변경 없음, 배포 관련 변경 없음.

### 0. ✅ 추가 업데이트 (2026-02-20, Codex 반영)

아래 내용은 기존 HANDOFF 이후 추가된 실제 변경(diff) 기준으로 반영:

- 백엔드 실행 설정 외부화:
`backend/app/config.py`에 `host`, `port` 설정 추가.
`backend/run_server.py`에 `--host`, `--port` CLI 인자 추가(기본값은 settings/환경변수).
- SOC 안내 UI 공통 컴포넌트화:
`frontend/src/components/common/SOCLowInfoTooltip.tsx` 신규 추가.
`Demo.tsx`, `Results.tsx`에서 중복 Tooltip 구현 제거 후 공통 컴포넌트 사용.
- SOH 게이지 상태 기준 변경:
`frontend/src/components/battery/CircularSOHGauge.tsx`를 ACC II 워런티 하한 기준으로 갱신.
`>=75` Healthy(초록), `70-74.9` Warning(노랑), `<70` Critical(빨강).
- Solar 충전 애니메이션 리팩터링:
`frontend/src/components/animations/SolarChargingAnimation.tsx`가
Framer Motion 기반 상태루프에서 GSAP timeline 기반 3단계 루프로 교체됨
(charging -> converting -> selling), 캔버스 크기 `340x500`으로 확장.
- 프론트엔드 의존성 추가:
`frontend/package.json` 및 lockfile에 `gsap` 추가.
- 랜딩 카피 미세 조정:
`frontend/src/i18n/locales/en/pages.json`에서 `titleLine1`을
`SolarX: AI-Powered` -> `SolarX: AI Powered`로 수정.

### 1. ✅ BatteryGauge SOC 툴팁 개선

**문제:** 각 배터리 게이지마다 "왜 SOC가 낮나요?" 툴팁이 개별 표시됨 → 중복, 지저분.

**해결:**
- `SOC_LOW_THRESHOLD = 50` 기준으로, 영역 내 배터리가 하나라도 SOC가 낮으면 영역 우측 상단에 툴팁 **하나만** 표시
- `BatteryGauge.tsx`: `showExplanation` prop 제거, `SOC_EXPLANATION`과 `SOC_LOW_THRESHOLD` export 추가
- `Demo.tsx`: Battery Status 박스에 `hasLowSOC` 조건 추가, 우측 상단 절대 위치 툴팁
- `Results.tsx`: Winner 카드 4번째 셀에 `position="relative"` + 조건부 툴팁

**수정 파일:**
- `frontend/src/components/battery/BatteryGauge.tsx`
- `frontend/src/pages/Demo.tsx`
- `frontend/src/pages/Results.tsx`

---

### 2. ✅ 차트 Y축/X축 레이블 전면 개선

**문제 4가지:**
1. Y축 레이블이 90° 회전되어 읽기 불편
2. Y축 숫자(₩10,000,000 같은 긴 값)가 왼쪽으로 잘림
3. X축/Y축 레이블이 tick 숫자와 겹침
4. X축 레이블 "시간(시간)" — 단위 표기 부정확

**해결:**
- Y축 레이블: Recharts `label` prop 완전 제거 → 차트 **위쪽**에 Chakra `<Text>`로 가로 렌더링
- X축 레이블: 차트 **아래쪽**에 Chakra `<Text>`로 분리 (겹침 없음)
- Y축 너비: `YAxis width` 증가 — ProfitChart `115px`, PredictionChart `65px`
- `labelPaddingLeft = margin.left + Y_AXIS_WIDTH`로 레이블 위치 정렬

**단위 변경:**
- **ProfitChart** X축: hours → days 변환 (`Math.round(hour / 24)`) / 레이블 `"일 (day)"` / 툴팁 `"{{day}}일차"`
- **PredictionChart** X축: 그대로 hours / 레이블 `"시간 (h)"` / 툴팁 `"{{hour}}h"`
- Brush도 day 단위로 표시

**수정 파일:**
- `frontend/src/components/charts/ProfitChart.tsx`
- `frontend/src/components/charts/PredictionChart.tsx`
- `frontend/src/i18n/locales/ko/charts.json`
- `frontend/src/i18n/locales/en/charts.json`

---

### 3. ✅ SolarChargingAnimation 하단 텍스트 개선

**변경 내용:**

| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| fontSize | 9 | 12 |
| fill | #6b7280 (어두운 회색) | #d1d5db (밝은 회색) |
| fontWeight | (없음) | bold |
| letterSpacing | 3 | 4 |
| text y 위치 | 372 | 384 (배터리와 간격 확보) |
| SVG height | 380 | 410 (하단 여백 확보) |
| SVG viewBox | 0 0 260 380 | 0 0 260 410 |

- 배터리 하단(y=356)에서 텍스트 중심(y=384)까지: **28px 간격**
- 텍스트 하단(y≈390)에서 SVG 테두리(410)까지: **20px 여백**

**수정 파일:**
- `frontend/src/components/animations/SolarChargingAnimation.tsx`

---

### 4. ✅ 홈페이지 메인 타이틀 줄바꿈

**목표:**
```
SolarX: AI 기반
배터리 최적화
```

**시도한 방법들 (모두 실패):**

| 방법 | 결과 | 실패 이유 |
|------|------|-----------|
| `\n` + `whiteSpace="pre-line"` | ❌ | CSS 자연 줄바꿈이 `\n` 보다 먼저 작동, "기반" 분리됨 |
| JSON에 실제 newline 삽입 | ❌ | JSON 파싱 오류 (Invalid control character) |
| `<br/>` + `Trans` 컴포넌트 | ❌ | `<br/>` 이전에 자연 줄바꿈 발생, "기반" 여전히 분리됨 |
| `wordBreak="keep-all"` | ❌ | 폰트가 너무 커서 컨테이너 폭 초과, 여전히 분리 |

**최종 성공 방법:**
번역 키를 두 줄로 분리 + 첫 번째 줄에 `whiteSpace="nowrap"`:

```tsx
// Landing.tsx
<Box as="span" display="block" whiteSpace="nowrap">
  {t('pages:landing.hero.titleLine1')}  // "SolarX: AI 기반"
</Box>
<Box as="span" display="block">
  {t('pages:landing.hero.titleLine2')}  // "배터리 최적화"
</Box>
```

`whiteSpace="nowrap"`이 CSS 자연 줄바꿈 자체를 차단하므로 화면 폭에 무관하게 첫 줄은 절대 깨지지 않음.

**부제목도 변경:**
- 줄바꿈 위치: "을 위한" 다음에서 줄바꿈 (`<br/>` + `Trans`)
- 브랜드명 통일: 삼성→Samsung, 테슬라→Tesla (영문)

**수정 파일:**
- `frontend/src/pages/Landing.tsx`
- `frontend/src/i18n/locales/ko/pages.json` (`titleLine1`, `titleLine2` 추가)
- `frontend/src/i18n/locales/en/pages.json` (`titleLine1`, `titleLine2` 추가)

---

### 5. ✅ 부수 오류 수정

이번 세션 작업 중 발생한 빌드/타입 오류들:

| 오류 | 원인 | 해결 |
|------|------|------|
| `TS1002: Unterminated string literal` (JSON) | Python Write tool이 `\n`을 실제 newline 바이트로 삽입 | json.dumps로 재작성 |
| `TS6133: 'AnimatePresence' declared but never read` | SolarChargingAnimation.tsx import 불필요 | import 제거 |
| `TS2322: labelFormatter type mismatch` | `(label: number)` → Recharts expects `ReactNode` | 타입 annotation 제거 |

---

## 📍 현재 프론트엔드 구조 (UI 관련 핵심 파일)

```
frontend/src/
├── components/
│   ├── animations/
│   │   └── SolarChargingAnimation.tsx   ✅ 수정됨 (SVG 높이, 텍스트 크기)
│   ├── battery/
│   │   ├── BatteryGauge.tsx             ✅ 수정됨 (showExplanation 제거, export 추가)
│   │   └── CircularSOHGauge.tsx
│   └── charts/
│       ├── ProfitChart.tsx              ✅ 수정됨 (축 레이블, days 변환)
│       ├── PredictionChart.tsx          ✅ 수정됨 (축 레이블)
│       └── MethodologyPanel.tsx
├── i18n/locales/
│   ├── ko/
│   │   ├── charts.json                  ✅ 수정됨 (xAxis 단위, tooltip)
│   │   └── pages.json                   ✅ 수정됨 (titleLine1/2, subtitle br)
│   └── en/
│       ├── charts.json                  ✅ 수정됨
│       └── pages.json                   ✅ 수정됨
└── pages/
    ├── Landing.tsx                      ✅ 수정됨 (Trans, nowrap, titleLine1/2)
    ├── Demo.tsx                         ✅ 수정됨 (단일 SOC 툴팁)
    └── Results.tsx                      ✅ 수정됨 (단일 SOC 툴팁)
```

---

## ✅ 전체 완료된 작업 이력

### 2026-02-20 (이번 세션)
- ✅ BatteryGauge 툴팁 단일화 (영역당 하나)
- ✅ 차트 Y축 레이블 회전 제거, 가로 배치
- ✅ 차트 Y축 숫자 잘림 해결
- ✅ 차트 X축 단위 수정 (시간→일, h 표기)
- ✅ CHARGING/CONVERTING/SELLING 텍스트 크게, 밝게
- ✅ 홈 타이틀 줄바꿈 ("SolarX: AI 기반" / "배터리 최적화")
- ✅ 홈 부제목 줄바꿈 및 브랜드명 영문 통일

### 2026-02-17 (이전 세션)
- ✅ GitHub Actions 수정 (디렉토리 구조, Docker load)
- ✅ Docker 컨테이너화 완료
- ✅ Backend API 구현 (FastAPI + LSTM)
- ✅ Frontend UI 구현 (React + Vite)
- ✅ CI/CD 파이프라인 구성
- ✅ 문서화 (QUICK_START.md, DOCKER_DEPLOYMENT.md 등)

---

## ⚠️ 알려진 미완료 사항

### 클라우드 배포 ⚠️
**모든 기술적 준비는 완료됨.** 명령만 실행하면 됨.

```bash
# 옵션 A: Google Cloud Run (권장)
gcloud run deploy solarx \
  --image ghcr.io/iimmuunnee/solarx:latest \
  --platform managed --region us-central1 \
  --allow-unauthenticated --memory 2Gi --port 8000

# 옵션 B: Fly.io
fly launch && fly deploy

# 옵션 C: Heroku
heroku create && heroku container:push web && heroku container:release web
```

---

## 🚀 다음 에이전트가 해야 할 작업

### 우선순위 1 🔴 — 클라우드 배포
현재 상태: Docker 빌드 ✅, GitHub Actions ✅, 로컬 테스트 ✅, 배포만 안 됨.

### 우선순위 2 🟡 — 추가 UI 개선 (선택)
아직 손대지 않은 영역:
- 모바일 반응형 검토 (특히 홈 타이틀이 `nowrap`으로 좁은 화면에서 넘칠 수 있음)
- Demo 페이지 결과 테이블 미구현 (현재 MetricsGrid만 있음)
- Architecture 페이지 실제 다이어그램 구현

### 우선순위 3 🟢 — 프로덕션 보안
- `backend/app/config.py` CORS `"*"` → 실제 도메인으로 변경
- 환경변수 `.env` 분리

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

1. **홈 타이틀 모바일** — `whiteSpace="nowrap"`이 적용된 "SolarX: AI 기반"은 화면 폭이 매우 좁으면 컨테이너를 초과할 수 있음. 현재 `base: '5xl'`로 모바일 폰트 크기를 줄여 대응 중이나, 아주 작은 화면에서는 테스트 필요.
2. **BatteryGauge SOC 툴팁 threshold** — `SOC_LOW_THRESHOLD = 50` 으로 설정. `BatteryGauge.tsx`에서 export 되어 있으므로 필요 시 변경 가능.
3. **ProfitChart Brush** — 줌 Brush의 범위 레이블도 day 단위로 표시됨. 툴팁은 "{{day}}일차" 포맷.
4. **JSON 수정 주의** — 이 프로젝트의 locale JSON 파일을 수정할 때 Python으로 직접 바이트를 다뤄야 함. Write 도구는 `\n`을 실제 newline으로 삽입하여 JSON 파싱 오류를 일으킴. `json.dumps()`를 사용해 재직성할 것.

---

**마지막 업데이트:** 2026-02-20 01:10
**다음 작업:** 클라우드 배포
**빌드 상태:** ✅ `npm run build` 통과 (오류 없음)

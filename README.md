# SolarX — 태양광 발전·ESS 운영 경제성 시뮬레이터

> LSTM 발전량 예측 + 물리 기반 배터리 모델로 ESS 운영 전략의 경제성을 시뮬레이션하고,
> **그 결과를 실제 전력시장 데이터(KPX SMP)로 다시 검증한** 개인 프로젝트입니다.

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-EE4C2C?logo=pytorch&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-23%20passed-0A9EDC?logo=pytest&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-Multi--Stage-2496ED?logo=docker&logoColor=white)

**Live Demo**: [https://solarx-5pbh.onrender.com](https://solarx-5pbh.onrender.com)

---

## 🔍 무엇을 검증했나

이 프로젝트에서 가장 의미 있는 결과는 높은 수익률이 아니라, **시뮬레이션을 의심하고 검증하는 과정**이었습니다.

1. **주간 수익 20억 원이 나왔을 때 축하 대신 원인을 찾았다** — 공공데이터 발전량의 단위 표기 문제로 값이 1,000배 증폭돼 있었다. → [Case 1: 단위 정합 버그](#case-1-비현실적인-수익-데이터--단위가-1000배-틀려-있었다)
2. **가상 가격으로 +26%였던 수익률이, KPX 실측 SMP(2024~2025)를 넣자 +8.31%로 떨어졌다** — 낙관 편향을 실데이터로 정량 확인하고, 줄어든 수치를 그대로 보고했다. → [핵심 결과](#2--핵심-결과-kpx-실측-데이터-기준)
3. **"돌아간다"가 아니라 "물리 법칙을 지킨다"를 테스트로 고정했다** — 배터리 충·방전 물리, 데이터 검증, 모델 구조를 [pytest 23건](#4--테스트--검증)으로 검증한다.

---

## 📅 프로젝트 정보

- **진행 기간**: 2026.01 ~ 2026.02 · **개인 프로젝트 (1인)**
- **담당 영역**:
  - **데이터 파이프라인**: 공공데이터 수집·전처리(단위 검증 포함), KPX SMP 가격 데이터 연동
  - **모델링**: PyTorch LSTM 발전량 예측 (early stopping, validation split)
  - **시뮬레이션**: 물리 제약(C-rate·DoD·효율·SOH·온도) 기반 ESS 운영 시뮬레이터, 배터리 3사 스펙 비교
  - **서비스화**: FastAPI + React 웹 앱, Docker Multi-stage 빌드, GitHub Actions → Render 배포

> **코드 위치**: 핵심 로직은 [`src/`](src), 검증은 [`tests/`](tests), 웹 서비스는 [`backend/`](backend)·[`frontend/`](frontend).
> 루트의 `SolarX_v*.ipynb`는 초기 탐색용 아카이브입니다.

---

## 1. 프로젝트 개요

**왜 시작했나** — 휴머노이드 로봇이 공장 인력을 대체하기 시작하면, 로봇을 위한 충전 인프라가 필요해집니다. 낮에는 로봇이 일하는 동안 태양광으로 배터리를 채우고, 가득 차면 남는 전기를 계통에 팔아 수익을 내고, 밤에는 로봇이 그 배터리로 충전하는 스테이션 — 이 시스템이 실제로 수지가 맞는지 숫자로 확인해보고 싶었습니다. 태양광 발전량은 날씨에 좌우되므로 기상 데이터로 LSTM을 학습시켰고, 그 예측으로 충전과 판매를 관리하는 시뮬레이터를 만들었습니다.

태양광 발전소에 ESS(에너지 저장 장치)를 붙였을 때, **언제 충전하고 언제 판매하는 것이 경제적으로 최적인지**를 시뮬레이션합니다.

- **예측**: 기상 데이터(일사량·기온·습도 등)로 다음 시간대 발전량을 LSTM으로 예측
- **운영 전략**: 예측 발전량과 시장 가격(SMP)에 따라 충전/방전/직판매를 결정하는 rule-based 로직
- **배터리 모델**: 제조사별 스펙(효율·C-rate·열화율)을 반영한 물리 기반 시뮬레이션 — LG에너지솔루션(2.0C/98.0%) · 삼성SDI(1.8C/98.5%) · Tesla 4680(1.5C/97.0%)
- **적용 시나리오**: 휴머노이드 로봇 충전 스테이션을 가정한 46파이 원통형 배터리 비교 (시나리오 설정이며, 로직 자체는 일반 ESS에 동일하게 적용됩니다)

## 2. 📊 핵심 결과 (KPX 실측 데이터 기준)

**실제 전력거래소(KPX)의 2024~2025년 SMP 데이터**를 적용한 시뮬레이션 결과입니다.

![Benchmark Graph](./images/benchmark_graph.png)

| Rank | 제조사 | 누적 수익 | 개선율 | 해석 |
|:---|:---|:---|:---|:---|
| 🥇 | **Samsung SDI** (NCA) | 약 6,160만 원 | **+8.31%** | 최고 효율(98.5%)로 왕복 손실 최소화 |
| 🥈 | LG Energy Solution (NCM) | 약 6,150만 원 | +8.15% | 고출력(2.0C)이나 효율 차이로 근소한 2위 |
| 🥉 | Tesla In-house (4680) | 약 6,080만 원 | +6.97% | 표준형 모델, 안정적이나 효율 열세 |
| — | Baseline (ESS 없음) | 약 5,690만 원 | 0.00% | 비교 기준점 (직판매만) |

> 수치는 시뮬레이션 가정(설비 용량, 기간, SMP 2024~2025)에 종속적이므로 유효숫자를 줄여 표기했습니다. 원 단위 수치는 [실행 결과](#6--실행-방법)에서 재현할 수 있습니다.

**이 결과에서 배운 것 두 가지:**

- **가상 대비 실측의 간극**: 가상의 고정 가격으로는 +26% 수익률이 나왔지만, 실제 SMP의 제한적인 가격 변동폭 안에서는 +8.31%로 줄었다. 시뮬레이션의 낙관 편향은 실데이터를 넣기 전까지는 보이지 않는다.
- **현실 시장에서 유효한 전략**: 가격 변동폭이 제한적인 환경에서는 "대박"을 노리는 전략보다 **손실을 허용하지 않는 고효율 전략**이 이긴다 — 0.5%p의 효율 차이가 순이익 순위를 갈랐다.

### 발전량 예측 성능

![Prediction Graph](./images/prediction_graph.png)

- 평가 지표: MAE 53.78 kW · RMSE 92.56 kW (테스트 구간)
- MAPE는 야간 발전량 0 구간이 많은 태양광 특성상 과대 계상되어 참고 지표로만 사용

## 3. 🔧 트러블슈팅

### Case 1: 비현실적인 수익 데이터 — 단위가 1,000배 틀려 있었다

- **증상**: 초기 시뮬레이션에서 주간 수익이 20억 원으로 산출됐다. 코드는 에러 없이 끝까지 돌았고, 그래프도 그럴듯했다 — 규모만 비상식적이었다.
- **추적**: 수익 계산을 역산하며 발전량 데이터의 원천을 확인했다. 공공데이터의 발전량 컬럼이 `kW`가 아니라 `Wh` 단위로 기록되어 있어, 파이프라인 전체에서 값이 1,000배 증폭된 상태였다.
- **해결**: 시뮬레이션 입력 단계에서 단위 변환(`Wh → kW`)을 명시적으로 적용하고, 이후 모든 에너지 흐름을 kW(전력)·kWh(에너지)로 통일했다. 시간 간격도 암묵적 1시간 가정 대신 `dt_hours` 파라미터로 명시했다. → [`main.py`](main.py) · [`simulation_service.py`](backend/app/services/simulation_service.py)
- **검증**: 변환 이후의 에너지 흐름은 배터리 물리 테스트(에너지 보존·C-rate·SoC 제약)로 고정했다(→ [`tests/unit/test_battery.py`](tests/unit/test_battery.py)). 다만 단위 변환 지점 자체를 고정하는 회귀 테스트는 아직 없다 — 변환 로직이 두 곳(CLI·웹 백엔드)에 중복돼 있어, 공용 함수로 추출한 뒤 테스트를 붙이는 것을 개선 과제로 남겼다.
- **교훈**: "결과가 이상하게 좋다"는 버그의 신호다. 외부 데이터는 단위 명세부터 의심해야 한다.

### Case 2: 잉여 전력이 조용히 버려지고 있었다 — Bypass Logic

- **증상**: 발전량이 큰 시간대에 수익이 기대보다 낮았다. 에러는 없었다.
- **추적**: 충전 분기를 확인하니 배터리 C-rate 제한으로 흡수하지 못한 잉여 전력이 어디로도 가지 않고 **소멸**하고 있었다 — `if action == charge` 분기가 배터리에 들어간 양만 계산하고 나머지를 처리하지 않았다.
- **해결**: **Bypass Logic**을 추가해 배터리가 감당하지 못하는 전력을 즉시 계통에 직판매하도록 했다 (`거래량 = 발전량 + 배터리 변화량` 에너지 보존식 기준). → [`src/battery.py`](src/battery.py)
- **검증**: 충·방전이 C-rate·SoC 제약을 지키는지, 에너지가 소멸하지 않는지를 배터리 물리 테스트로 고정했다. → [`tests/unit/test_battery.py`](tests/unit/test_battery.py)

### Case 3: 상향 평준화된 성능 차이를 어떻게 보여줄 것인가

- **문제**: 3사 배터리 성능이 비슷해 단순 누적 그래프로는 차이가 보이지 않았다.
- **해결**: Baseline 대비 **순수 추가 수익(Net Profit Gain)** 그래프를 별도로 구현해, 0.5%의 효율 차이가 만드는 격차를 증폭해 시각화했다.

## 4. 🧪 테스트 & 검증

```bash
pytest tests/ -v --cov=src        # 23 passed
```

| 분류 | 파일 | 건수 | 검증 내용 |
|:---|:---|:---|:---|
| 단위 | `tests/unit/test_battery.py` | 8 | 충·방전 물리(C-rate·SoC·DoD 제약), 효율 적용, 과방전 방지 |
| 단위 | `tests/unit/test_data_loader.py` | 5 | 시퀀스 생성·시계열 순서 보존, 스케일러 일관성, 기상 컬럼 정규화, 빈 데이터 처리 |
| 단위 | `tests/unit/test_model.py` | 6 | LSTM forward·shape, 재현성, 배치 독립성, gradient flow |
| 통합 | `tests/integration/test_simulation_pipeline.py` | 4 | End-to-end 시뮬레이션 파이프라인 |

### 지역별 시나리오 검증 (Robustness)

특정 지역(동해) 데이터에 과적합되지 않는지, 발전량 계수를 조정한 스트레스 테스트로 확인했습니다.

![Scalability Graph](./images/scalability_graph.png)

| 시나리오 | 발전량 계수 | 결과 |
|:---|:---|:---|
| 동해 (기준) | 1.0 | 약 6,160만 원 — Baseline |
| 제주 (고일사) | 1.3 | 약 8,010만 원 — 발전량 급증 시에도 Bypass 로직이 잉여 전력 손실을 방지 |
| 시애틀 (저일사) | 0.6 | 약 3,700만 원 — 발전량 40% 감소에도 방전 제어로 수익성 방어 |

> 발전량 규모와 무관하게 수익 구조가 선형적으로 유지됨을 확인했습니다. 다만 이는 **발전량 계수를 조정한 가상 시나리오**로, 실제 기후 데이터 기반 검증은 향후 과제입니다.

## 5. ⚠️ 한계점 및 향후 과제

숫자를 좋게 만들기 위해 덮어둔 가정이 없도록, 모델이 다루지 못하는 것을 명시합니다.

- **배터리 모델의 '외적 타당성'은 미검증**: 물리 테스트는 모델이 자기 규칙(에너지 보존·C-rate·SoC 제약)을 지키는지 고정할 뿐이다. 모델 구조는 표준 ESS 시뮬레이션 관행(왕복 효율의 단방향 분해, C-rate 전력 제한)을 따르지만, 제조사별 파라미터(효율·열화율)는 공개 자료 기반 **가정**이며 실제 배터리와의 일치 여부는 도메인 지식 한계로 검증하지 못했다 — 데이터시트·문헌 대조가 다음 과제
- **로봇 충전 수요가 결정적(deterministic)**: 확률적 수요 모델링 미반영 — 서비스율(Service Level) 중심 전략으로 발전 필요
- **BTMS(열관리) 소비 전력 미포함**: 겨울 히팅·여름 쿨링 에너지가 수익을 잠식하는 효과 미반영
- **시간 해상도 1시간 고정**: 분 단위 가격 스파이크 대응 불가
- **rule-based 전략의 한계**: 임계값 기반 충·방전 결정 — 강화학습(DQN/PPO) 대체 여지
- 해결 완료된 항목(SOH 열화 모델링, 온도 효과, CAPEX/ROI/NPV 분석)은 [프로젝트 변천사](#7-프로젝트-변천사) 참고

## 6. 💻 실행 방법

### Docker (권장)

```bash
git clone https://github.com/iimmuunnee/SolarX && cd SolarX
docker compose up --build
# → http://localhost:8000
```

### Python CLI (시뮬레이션 직접 실행)

```bash
pip install -r requirements.txt
python src/train.py     # LSTM 학습 (early stopping)
python main.py          # 벤치마크 + 지역별 시나리오 시뮬레이션
pytest tests/ -v        # 테스트 23건
```

### 로컬 개발 (웹 앱)

```bash
cd backend && pip install -r requirements.txt && python run_server.py   # :8000
cd frontend && npm install && npm run dev                               # :5173
```

## 7. 프로젝트 변천사

| 버전 | 내용 |
|:---|:---|
| v1.x | LSTM 발전량 예측 + 방전 제어·단위 일관화 등 초기 신뢰성 수정 |
| v2.0 | 배터리 물리 제약(C-rate·DoD·효율) 반영 |
| v3.0 | 휴머노이드 충전 스테이션·46파이 배터리 시나리오로 구체화 |
| v4.x | **KPX 실측 SMP(2024~2025) 연동** — 가상 가격 제거, MAE/RMSE 평가 정착 |
| v5.0 | pytest 테스트 스위트(23건) · SOH 열화 · 온도 효과 · CAPEX/ROI/NPV 분석 |
| v6.0 | React + FastAPI 웹 앱, Docker Multi-stage, GitHub Actions → Render CI/CD |

## 8. 시스템 구성

```
data/ (기상·발전량·SMP CSV)
  → src/data_loader.py   데이터 검증·단위 변환·시퀀스 생성
  → src/model.py·train.py  LSTM 예측 (PyTorch)
  → src/battery.py       물리 기반 배터리 모델 (3사 스펙·SOH·온도)
  → main.py              운영 전략 시뮬레이션·수익 계산
  → src/economics.py     CAPEX/OPEX/ROI/NPV
  → backend/ (FastAPI)   시뮬레이션 API (/api/simulate/benchmark 등)
  → frontend/ (React 19) 인터랙티브 대시보드 (i18n KO/EN)
```

| 웹 API | 설명 |
|:---|:---|
| `POST /api/simulate/benchmark` | 3사 비교 벤치마크 실행 |
| `POST /api/simulate/custom` | 단일 제조사 커스텀 시뮬레이션 |
| `GET /api/results/scalability` | 지역별 시나리오 결과 조회 |
| `GET /api/health` | 상태 확인 |

## 9. 참고 자료

- Tesla Battery Day (2020) — 4680 탭리스 구조, 옵티머스 소비 전력 시나리오
- Samsung SDI InterBattery 2024/2025 — 46파이 원통형 로드맵, NCA(PRiMX)
- LG Energy Solution Tech Conference — 고출력 원통형 셀 급속 충전 프로파일
- 배터리 물리: Joule Heating(P=I²R), Butler-Volmer 방정식 기반 충·방전 모델링

---

**개인 프로젝트** · 임휘훈 ([@iimmuunnee](https://github.com/iimmuunnee)) · 2026.01 ~ 2026.02

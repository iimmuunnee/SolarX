# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SolarX is a battery optimization system for solar-powered humanoid robot charging stations. It compares three global battery vendors (LG Energy Solution, Samsung SDI, Tesla) using LSTM-based solar generation prediction and physics-informed battery models to determine optimal economic performance.

**Core Technologies**: Python, PyTorch, LSTM time series prediction, energy storage simulation

### 한국어

SolarX는 태양광 기반 휴머노이드 로봇 충전소를 위한 배터리 최적화 시스템입니다. LSTM 기반 태양광 발전량 예측과 물리 기반 배터리 모델을 사용하여 세 개의 글로벌 배터리 제조사(LG Energy Solution, Samsung SDI, Tesla)를 비교하고 최적의 경제적 성능을 결정합니다.

**핵심 기술**: Python, PyTorch, LSTM 시계열 예측(time series prediction), 에너지 저장 시뮬레이션(energy storage simulation)

## Development Commands

### Setup (Python Simulation)
```bash
pip install -r requirements.txt
```

### Training the LSTM Model
```bash
python src/train.py
```
- Trains LSTM model on historical weather and solar generation data
- Saves model to `src/lstm_solar_model.pth`
- Uses 24-hour sequence length, 100 epochs
- Time series shuffle is disabled (shuffle=False) to preserve temporal order

### Running Simulations (CLI)
```bash
python main.py
```
- Runs two simulations:
  1. **Part 1**: Global battery benchmark (LG vs Samsung vs Tesla)
  2. **Part 2**: Scalability test (Donghae base, Jeju high solar, Seattle low solar)
- Generates plots in `./images/` directory
- Outputs profit metrics with MAE/RMSE/MAPE for prediction accuracy

## Web Application Deployment

### Docker (Recommended for Production)

The web application is containerized using Docker with a multi-stage build that bundles both the React frontend and FastAPI backend into a single container.

**Build Docker Image:**
```bash
docker build -t solarx-web .
```

**Run Docker Container:**
```bash
docker run -p 8000:8000 solarx-web
```

**Using docker-compose (Recommended for local development):**
```bash
docker-compose up
```

**Stop docker-compose:**
```bash
docker-compose down
```

**Access the Application:**
- Web UI: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/health

### Local Development (Without Docker)

**Backend Development:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend Development:**
```bash
cd frontend
npm install
npm run dev
# Frontend dev server runs on http://localhost:5173
```

**Note:** When running locally without Docker, update `frontend/.env` to point to backend:
```
VITE_API_URL=http://localhost:8000
```

### Environment Variables

The application can be configured using environment variables:

- `PYTHONUNBUFFERED=1` - Enable real-time log output
- `PORT` - Override default port 8000 (some cloud platforms require this)

### Deployment to Cloud Platforms

**Google Cloud Run:**
```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/solarx-web

# Deploy to Cloud Run
gcloud run deploy solarx \
  --image gcr.io/YOUR_PROJECT_ID/solarx-web \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1
```

**Fly.io:**
```bash
# Install flyctl
fly launch

# Deploy
fly deploy
```

**Heroku:**
```bash
# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Push container
heroku container:push web
heroku container:release web
```

### Docker Architecture

The Dockerfile uses a multi-stage build:

1. **Stage 1 (frontend-build)**:
   - Base: `node:18-alpine`
   - Builds React frontend with Vite
   - Output: Optimized static files in `dist/`

2. **Stage 2 (python runtime)**:
   - Base: `python:3.9-slim`
   - Installs Python dependencies
   - Copies SolarX modules and backend code
   - Copies frontend build from Stage 1
   - Serves both API and static files on port 8000

**Key Features:**
- Single container deployment
- Production-optimized build (~800MB - 1.2GB image size)
- Health checks for container orchestration
- SPA routing support (refresh on /demo works correctly)
- API documentation at /docs (FastAPI auto-generated)

### 한국어: 개발 명령어

#### 설치
```bash
pip install -r requirements.txt
```

#### LSTM 모델 학습
```bash
python src/train.py
```
- 과거 기상 데이터와 태양광 발전량 데이터로 LSTM 모델을 학습합니다
- 모델을 `src/lstm_solar_model.pth`에 저장합니다
- 24시간 시퀀스 길이(sequence length), 100 에포크(epochs)를 사용합니다
- 시계열 순서를 보존하기 위해 시계열 셔플이 비활성화됩니다 (shuffle=False)

#### 시뮬레이션 실행
```bash
python main.py
```
- 두 가지 시뮬레이션을 실행합니다:
  1. **Part 1**: 글로벌 배터리 벤치마크 (LG vs Samsung vs Tesla)
  2. **Part 2**: 확장성 테스트 (동해 기준, 제주 고일사, Seattle 저일사)
- `./images/` 디렉토리에 그래프를 생성합니다
- 예측 정확도를 위한 MAE/RMSE/MAPE와 함께 수익 지표를 출력합니다

## Code Architecture

### Data Flow
1. **Data Loading** (`src/data_loader.py`):
   - Auto-detects CSV/Excel files in `data/` directory
   - Merges weather data (기온, 강수량, 풍속, 습도, 일조, 일사, 운량)
   - Integrates solar generation data (Wh → kW conversion applied)
   - Loads SMP (System Marginal Price) from Korean power exchange
   - Creates 24-hour sequences for LSTM input

2. **Prediction** (`src/model.py`):
   - LSTM architecture: input_size=8, hidden_size=64, num_layers=1
   - Predicts solar generation based on weather features
   - Uses Accelerate library for hardware optimization

3. **Battery Simulation** (`src/battery.py`):
   - **Base class**: `ESSBattery` with physics constraints (C-rate, efficiency, SoC limits)
   - **Vendor models**: LG (2.0C, 98.0%), Samsung (1.8C, 98.5%), Tesla (1.5C, 97.0%)
   - Efficiency is round-trip converted to one-way (sqrt applied)
   - Returns grid trade power: positive for discharge (selling), negative for charge (buying)

4. **Decision Logic** (`main.py`):
   - **Charge**: when price < avg_price * 0.9 and predicted generation > 0.1 kW
   - **Discharge**: when price > avg_price * 1.1
   - **Bypass Logic**: excess solar beyond battery C-rate limit is sold directly to grid
   - `allow_grid_charge` flag controls whether battery can charge from grid

### 한국어: 코드 아키텍처

#### 데이터 흐름(Data Flow)
1. **데이터 로딩** (`src/data_loader.py`):
   - `data/` 디렉토리에서 CSV/Excel 파일을 자동 감지합니다
   - 기상 데이터를 병합합니다 (기온, 강수량, 풍속, 습도, 일조, 일사, 운량)
   - 태양광 발전량 데이터를 통합합니다 (Wh → kW 변환 적용)
   - 한국 전력거래소의 SMP(System Marginal Price, 계통한계가격)를 로드합니다
   - LSTM 입력을 위한 24시간 시퀀스를 생성합니다

2. **예측(Prediction)** (`src/model.py`):
   - LSTM 아키텍처: input_size=8, hidden_size=64, num_layers=1
   - 기상 특성(weather features)을 기반으로 태양광 발전량을 예측합니다
   - 하드웨어 최적화를 위해 Accelerate 라이브러리를 사용합니다

3. **배터리 시뮬레이션** (`src/battery.py`):
   - **베이스 클래스**: 물리 제약조건(C-rate, 효율, SoC 한계)을 가진 `ESSBattery`
   - **제조사 모델**: LG (2.0C, 98.0%), Samsung (1.8C, 98.5%), Tesla (1.5C, 97.0%)
   - 왕복 효율(round-trip efficiency)이 단방향(one-way)으로 변환됩니다 (sqrt 적용)
   - 계통 거래 전력(grid trade power)을 반환합니다: 방전(selling)은 양수, 충전(buying)은 음수

4. **의사결정 로직(Decision Logic)** (`main.py`):
   - **충전(Charge)**: 가격 < 평균가격 * 0.9 이고 예측 발전량 > 0.1 kW일 때
   - **방전(Discharge)**: 가격 > 평균가격 * 1.1일 때
   - **우회 로직(Bypass Logic)**: 배터리 C-rate 한계를 초과하는 태양광 발전량은 계통에 직접 판매됩니다
   - `allow_grid_charge` 플래그는 배터리가 계통에서 충전할 수 있는지 제어합니다

### Key Files and Responsibilities

- **main.py**: Orchestrates benchmark and scalability tests, calculates profits
- **src/battery.py**: Battery physics (charge/discharge, efficiency, C-rate, DoD)
- **src/data_loader.py**: Data ingestion, preprocessing, train/test split, sequence creation
- **src/model.py**: LSTM model definition and prediction interface
- **src/train.py**: Training loop with Accelerate for hardware optimization
- **src/visualizer.py**: Matplotlib-based report generation (requires Malgun Gothic font for Korean)

### Critical Design Decisions

1. **Unit Consistency**: All energy flows use kW (power) and kWh (energy) with explicit `dt_hours` parameter (typically 1.0 for hourly data)

2. **Efficiency Application**: Round-trip efficiency is converted to one-way efficiency via `sqrt(eff)` and applied separately to charge and discharge operations

3. **Discharge Control**: `amount_kw` parameter limits discharge power to prevent over-discharge beyond requested amounts

4. **Bypass Logic**: When battery C-rate cannot absorb all solar generation, excess is immediately sold to grid to prevent losses

5. **No Grid Charge by Default**: Set `allow_grid_charge=False` in decision logic to prevent arbitrage using only predicted solar (more conservative)

#### 한국어: 주요 파일 및 책임

- **main.py**: 벤치마크 및 확장성 테스트를 조율하고, 수익을 계산합니다
- **src/battery.py**: 배터리 물리(충방전, 효율, C-rate, DoD - Depth of Discharge)
- **src/data_loader.py**: 데이터 수집, 전처리, 학습/테스트 분할, 시퀀스 생성
- **src/model.py**: LSTM 모델 정의 및 예측 인터페이스
- **src/train.py**: 하드웨어 최적화를 위한 Accelerate를 사용한 학습 루프
- **src/visualizer.py**: Matplotlib 기반 보고서 생성 (한글 표시를 위해 Malgun Gothic 폰트 필요)

#### 한국어: 핵심 설계 결정사항

1. **단위 일관성(Unit Consistency)**: 모든 에너지 흐름은 kW(전력)와 kWh(에너지)를 사용하며 명시적인 `dt_hours` 매개변수를 사용합니다 (일반적으로 시간별 데이터의 경우 1.0)

2. **효율 적용(Efficiency Application)**: 왕복 효율(round-trip efficiency)은 `sqrt(eff)`를 통해 단방향 효율(one-way efficiency)로 변환되며 충전 및 방전 작업에 개별적으로 적용됩니다

3. **방전 제어(Discharge Control)**: `amount_kw` 매개변수는 요청된 양을 초과하는 과방전을 방지하기 위해 방전 전력을 제한합니다

4. **우회 로직(Bypass Logic)**: 배터리 C-rate가 모든 태양광 발전량을 흡수할 수 없을 때, 초과분은 손실을 방지하기 위해 즉시 계통에 판매됩니다

5. **기본적으로 계통 충전 없음(No Grid Charge by Default)**: 의사결정 로직에서 `allow_grid_charge=False`로 설정하여 예측된 태양광만 사용하여 차익거래(arbitrage)를 방지합니다 (더 보수적인 접근)

## Data Requirements

Place in `data/` directory:
- Weather CSV/Excel files containing: 기온(℃), 강수량(mm), 풍속(m/s), 습도(%), 일조(hr), 일사(MJ/m2), 운량(10분위)
- Solar generation CSV with hourly columns (01시, 02시, ..., 24시)
- SMP files with naming pattern containing "smp" and columns (1h, 2h, ..., 24h)

The data loader auto-detects file types based on column names.

### 한국어: 데이터 요구사항

`data/` 디렉토리에 다음을 배치하세요:
- 다음 항목을 포함하는 기상 CSV/Excel 파일: 기온(℃), 강수량(mm), 풍속(m/s), 습도(%), 일조(hr), 일사(MJ/m2), 운량(10분위)
- 시간별 컬럼이 있는 태양광 발전량 CSV (01시, 02시, ..., 24시)
- "smp"를 포함하는 명명 패턴을 가진 SMP 파일 및 컬럼 (1h, 2h, ..., 24h)

데이터 로더는 컬럼 이름을 기반으로 파일 유형을 자동 감지합니다.

## Testing and Validation

- **Prediction Metrics**: MAE, RMSE, MAPE printed during simulation
- **Economic Validation**: Compare against baseline (no ESS) to verify net profit gain
- **Physics Validation**: Check that discharge respects `amount_kw` limits and SoC constraints

### 한국어: 테스트 및 검증

- **예측 지표(Prediction Metrics)**: 시뮬레이션 중 MAE, RMSE, MAPE가 출력됩니다
- **경제성 검증(Economic Validation)**: 베이스라인(ESS 없음)과 비교하여 순이익 증가를 확인합니다
- **물리 검증(Physics Validation)**: 방전이 `amount_kw` 한계 및 SoC 제약조건을 준수하는지 확인합니다

## Known Constraints and Assumptions

### ✅ Resolved in v2.0:
- ~~Battery degradation (SOH) is not modeled~~ → **IMPLEMENTED**: SOH tracking with cycle counting and calendar aging
- ~~Temperature effects on efficiency are constant~~ → **IMPLEMENTED**: Temperature-dependent efficiency adjustment
- ~~Initial investment (CAPEX) not included in profit calculations~~ → **IMPLEMENTED**: Full CAPEX/OPEX/ROI/NPV analysis

### 🔄 Remaining Constraints:
- Robot charging demand is not stochastic (future work: probabilistic demand modeling)
- Time step is fixed at 1 hour (sufficient for current scope)
- BTMS (Battery Thermal Management System) power consumption not included

### 한국어: 알려진 제약조건 및 가정사항

### ✅ v2.0에서 해결됨:
- ~~배터리 열화(SOH - State of Health)는 모델링되지 않았습니다~~ → **구현 완료**: 사이클 카운팅 및 캘린더 에이징 기반 SOH 추적
- ~~효율에 대한 온도 영향은 일정합니다~~ → **구현 완료**: 온도 의존적 효율 조정
- ~~초기 투자비용(CAPEX)은 수익 계산에 포함되지 않았습니다~~ → **구현 완료**: 완전한 CAPEX/OPEX/ROI/NPV 분석

### 🔄 남은 제약사항:
- 로봇 충전 수요는 확률적(stochastic)이지 않습니다 (향후 작업: 확률적 수요 모델링)
- 시간 단계(time step)는 1시간으로 고정되어 있습니다 (현재 범위에는 충분)
- BTMS (배터리 열관리 시스템) 소비 전력은 포함되지 않았습니다

## New Features in v2.0

### Testing Infrastructure
- **Unit Tests**: Battery physics, data validation, model architecture
- **Integration Tests**: End-to-end simulation pipeline
- **Coverage Target**: 80%+
- **Run Tests**: `pytest tests/ -v --cov=src`

### Configuration Management
- **config.py**: Centralized configuration for all parameters
- **ModelConfig**: LSTM hyperparameters (hidden_size, dropout, learning_rate)
- **BatteryConfig**: Vendor specifications
- **SimulationConfig**: Decision thresholds, grid charge settings
- **PathConfig**: File paths for data, models, outputs

### Battery Enhancements
- **SOH Tracking**: `battery.soh`, `battery.cycle_count`, `battery.total_throughput_kwh`
- **Temperature Effects**: `battery.update(..., temp_c=25.0)`
- **Vendor Differences**:
  - LG: degradation_rate=0.00008/cycle (best longevity)
  - Samsung: degradation_rate=0.00009/cycle
  - Tesla: degradation_rate=0.0001/cycle
- **Status Reporting**: `battery.get_status()` returns comprehensive metrics

### Economic Analysis
- **CAPEX Calculation**: `economics.VENDOR_COSTS[vendor].total_capex(capacity)`
- **ROI Metrics**: `calculate_roi(revenue, capex, opex, years)`
- **Outputs**: ROI%, payback_period, NPV (5% discount rate)

### Model Improvements
- **Early Stopping**: Patience=15, saves best model
- **Gradient Clipping**: max_norm=1.0 for stability
- **Layer Normalization**: Improves convergence
- **Dropout**: Configurable regularization
- **Validation Split**: Train/Val 80/20

### Logging System
- **Structured Logs**: Timestamp, level, module name
- **Usage**: `from src.logger import setup_logger; logger = setup_logger("module")`
- **Levels**: INFO, WARNING, ERROR
- **Output**: Console + optional file

### 한국어: v2.0의 새로운 기능

### 테스트 인프라
- **유닛 테스트**: 배터리 물리, 데이터 검증, 모델 아키텍처
- **통합 테스트**: End-to-end 시뮬레이션 파이프라인
- **커버리지 목표**: 80%+
- **테스트 실행**: `pytest tests/ -v --cov=src`

### 설정 관리
- **config.py**: 모든 파라미터의 중앙 집중식 설정
- **ModelConfig**: LSTM 하이퍼파라미터 (hidden_size, dropout, learning_rate)
- **BatteryConfig**: 제조사 사양
- **SimulationConfig**: 의사결정 임계값, 그리드 충전 설정
- **PathConfig**: 데이터, 모델, 출력 파일 경로

### 배터리 개선사항
- **SOH 추적**: `battery.soh`, `battery.cycle_count`, `battery.total_throughput_kwh`
- **온도 효과**: `battery.update(..., temp_c=25.0)`
- **제조사별 차이**:
  - LG: degradation_rate=0.00008/cycle (최고 수명)
  - Samsung: degradation_rate=0.00009/cycle
  - Tesla: degradation_rate=0.0001/cycle
- **상태 보고**: `battery.get_status()`로 종합 지표 반환

### 경제성 분석
- **CAPEX 계산**: `economics.VENDOR_COSTS[vendor].total_capex(capacity)`
- **ROI 지표**: `calculate_roi(revenue, capex, opex, years)`
- **출력**: ROI%, 회수기간, NPV (5% 할인율)

### 모델 개선사항
- **Early Stopping**: Patience=15, 최적 모델 자동 저장
- **Gradient Clipping**: max_norm=1.0으로 안정성 확보
- **Layer Normalization**: 수렴 속도 향상
- **Dropout**: 설정 가능한 정규화
- **Validation 분할**: Train/Val 80/20

### 로깅 시스템
- **구조화된 로그**: 타임스탬프, 레벨, 모듈 이름
- **사용법**: `from src.logger import setup_logger; logger = setup_logger("module")`
- **레벨**: INFO, WARNING, ERROR
- **출력**: 콘솔 + 선택적 파일

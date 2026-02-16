"""SolarX: Battery optimization system for solar-powered robot charging stations."""
import numpy as np
import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.battery import SamsungSDI, LGEnergySolution, TeslaBattery
from src.data_loader import SolarDataManager
from src.visualizer import ReportGenerator
from src.model import LSTMPredictor
from src.logger import setup_logger
from src.economics import VENDOR_COSTS, calculate_roi
from config import Config

# Setup logger
logger = setup_logger("solarx.main")


def run():
    """Main simulation function."""
    logger.info("=" * 60)
    logger.info("SolarX: Real Data 시뮬레이션")
    logger.info("=" * 60)

    # Load configuration
    config = Config()
    dt_hours = 1.0
    allow_grid_charge = config.simulation.allow_grid_charge

    # 1. 데이터 로드 (Data Load)
    loader = SolarDataManager()
    try:
        _, _, test_x, test_y, test_smp = loader.load_and_split_standard(
            os.path.join(BASE_DIR, "data")
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        return

    seq_length = config.model.seq_length
    X_test, y_test = loader.create_sequences(test_x, test_y, seq_length=seq_length)

    # Get test DataFrame for temperature data
    # We need to reconstruct it from test_x for temperature access
    test_df = pd.DataFrame(
        loader.scaler_x.inverse_transform(test_x),
        columns=["기온(℃)", "강수량(mm)", "풍속(m/s)", "습도(%)", "일조(hr)", "일사(MJ/m2)", "운량(10분위)", "발전량"]
    )

    # SMP 정렬 (Alignment)
    real_prices = test_smp[seq_length:] if test_smp is not None else None

    logger.info(f"AI 예측 실행 중... (Test: {len(X_test)} hours)")

    # 2. 모델 예측 (Model Predict)
    predictor = LSTMPredictor(
        os.path.join(BASE_DIR, "src", "lstm_solar_model.pth"),
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=0.0  # No dropout during inference
    )
    y_pred_scaled = predictor.predict(X_test)

    y_real_raw = loader.inverse_transform_y(y_test.reshape(-1, 1)).flatten()
    y_pred_raw = loader.inverse_transform_y(y_pred_scaled.reshape(-1, 1)).flatten()

    y_real_kw = np.maximum(y_real_raw / 1000.0, 0)
    y_pred_kw = np.maximum(y_pred_raw / 1000.0, 0)

    # 기본 예측 지표 (Metrics)
    mae = np.mean(np.abs(y_real_kw - y_pred_kw))
    rmse = np.sqrt(np.mean((y_real_kw - y_pred_kw) ** 2))
    mape_mask = y_real_kw > 1e-6
    mape = (
        np.mean(np.abs((y_real_kw[mape_mask] - y_pred_kw[mape_mask]) / y_real_kw[mape_mask]))
        * 100.0
        if np.any(mape_mask)
        else float("nan")
    )
    logger.info(f"Eval -> MAE: {mae:.4f} kW | RMSE: {rmse:.4f} kW | MAPE: {mape:.2f}%")

    # SMP 확인 (Check)
    if real_prices is None or np.sum(real_prices) == 0:
        logger.warning("Warning: SMP 데이터가 없어 임시 가격 곡선을 사용합니다.")
        real_prices = [100 if 10 <= i % 24 <= 16 else 200 for i in range(len(y_real_kw))]
    else:
        logger.info(f"Using real SMP (avg: {np.mean(real_prices):.1f})")

    # ---------------------------------------------------------
    # PART 1: 글로벌 배터리 비교 (Benchmark)
    # ---------------------------------------------------------
    logger.info("\n>>> [Part 1] 글로벌 배터리 비교 (Benchmark)")

    battery_capacity_kwh = np.max(y_real_kw) * dt_hours * 3
    batteries = [
        LGEnergySolution(battery_capacity_kwh),
        SamsungSDI(battery_capacity_kwh),
        TeslaBattery(battery_capacity_kwh),
    ]

    results = {}
    baseline_history = []

    # 기준선 (Baseline)
    base_profit = 0
    for t in range(len(y_real_kw)):
        base_profit += y_real_kw[t] * dt_hours * real_prices[t]
        baseline_history.append(base_profit)
    logger.info(f"0. Baseline (ESS 없음): {int(base_profit):,} KRW")

    avg_price = np.mean(real_prices)

    for batt in batteries:
        profit = 0
        history = []

        for t in range(len(y_real_kw)):
            gen_kw = y_real_kw[t]
            pred_kw = y_pred_kw[t]
            price = real_prices[t]

            # Get temperature data (aligned with sequences)
            temp_c = test_df.iloc[seq_length + t]["기온(℃)"] if seq_length + t < len(test_df) else 25.0

            action = 0
            if price > avg_price * config.simulation.discharge_threshold:
                action = -1
            elif price < avg_price * config.simulation.charge_threshold and pred_kw > config.simulation.min_generation_kw:
                action = 1

            if action == 1:
                charge_request_kw = batt.max_power if allow_grid_charge else min(gen_kw, batt.max_power)
                trade_kw = gen_kw + batt.update(action, charge_request_kw, dt_hours, temp_c=temp_c)
            elif action == -1:
                discharge_request_kw = batt.max_power
                trade_kw = gen_kw + batt.update(action, discharge_request_kw, dt_hours, temp_c=temp_c)
            else:
                trade_kw = gen_kw

            profit += trade_kw * dt_hours * price
            history.append(profit)

        results[batt.name] = history

        # Get battery status
        status = batt.get_status()

        # CAPEX 분석 추가
        cost_model = VENDOR_COSTS[batt.name]
        capex = cost_model.total_capex(battery_capacity_kwh)

        # Assume simulation covers 1 year equivalent (adjust based on actual data)
        # For demonstration, we'll use the data period as-is
        simulation_years = len(y_real_kw) / (24 * 365)  # Convert hours to years

        roi_metrics = calculate_roi(
            total_revenue=profit,
            capex=capex,
            opex_annual=cost_model.ohm_cost_per_year,
            years=max(simulation_years, 1)  # At least 1 year for calculation
        )

        logger.info(f"\n{batt.name}:")
        logger.info(f"  Revenue: {int(profit):,} KRW")
        logger.info(f"  SOH: {status['soh']:.2%} (Cycles: {status['cycle_count']:.1f})")
        logger.info(f"  Throughput: {status['total_throughput_kwh']:.1f} kWh")
        logger.info(f"  CAPEX: ${capex:,.0f}")
        logger.info(f"  OPEX (annual): ${cost_model.ohm_cost_per_year:,.0f}")
        logger.info(f"  ROI: {roi_metrics['roi_percent']:.2f}%")
        logger.info(f"  Payback: {roi_metrics['payback_period_years']:.1f} years")
        logger.info(f"  NPV: ${roi_metrics['npv']:,.0f}")

    ReportGenerator.save_plots(y_real_kw, y_pred_kw, results, baseline_history)

    # ---------------------------------------------------------
    # PART 2: 확장성 테스트 (Scalability)
    # ---------------------------------------------------------
    logger.info("\n>>> [Part 2] 확장성 테스트 (Scalability)")

    scenarios = [
        ("Donghae (Base)", 1.0),
        ("Jeju (High Solar)", 1.3),
        ("Seattle (Low Solar)", 0.6),
    ]

    scalability_results = {}

    for name, factor in scenarios:
        scenario_gen_kw = y_real_kw * factor
        test_batt = SamsungSDI(battery_capacity_kwh * factor)

        profit = 0
        history = []

        for t in range(len(scenario_gen_kw)):
            gen_kw = scenario_gen_kw[t]
            pred_kw = y_pred_kw[t] * factor
            price = real_prices[t]

            # Get temperature
            temp_c = test_df.iloc[seq_length + t]["기온(℃)"] if seq_length + t < len(test_df) else 25.0

            action = 0
            if price > avg_price * config.simulation.discharge_threshold:
                action = -1
            elif price < avg_price * config.simulation.charge_threshold and pred_kw > config.simulation.min_generation_kw:
                action = 1

            if action == 1:
                charge_request_kw = test_batt.max_power if allow_grid_charge else min(gen_kw, test_batt.max_power)
                trade_kw = gen_kw + test_batt.update(action, charge_request_kw, dt_hours, temp_c=temp_c)
            elif action == -1:
                discharge_request_kw = test_batt.max_power
                trade_kw = gen_kw + test_batt.update(action, discharge_request_kw, dt_hours, temp_c=temp_c)
            else:
                trade_kw = gen_kw

            profit += trade_kw * dt_hours * price
            history.append(profit)

        scalability_results[name] = history

        # Log battery status
        status = test_batt.get_status()
        logger.info(f"  {name}: Final Profit {int(profit):,} KRW (SOH: {status['soh']:.2%})")

    ReportGenerator.save_scalability_plot(scalability_results)

    logger.info("\n" + "=" * 60)
    logger.info("시뮬레이션 완료! 결과는 images/ 디렉토리에 저장되었습니다.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()

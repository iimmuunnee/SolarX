import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.battery import SamsungSDI, LGEnergySolution, TeslaBattery
from src.data_loader import SolarDataManager
from src.visualizer import ReportGenerator
from src.model import LSTMPredictor


def run():
    print("=" * 60)
    print("SolarX: Real Data 시뮬레이션")
    print("=" * 60)

    dt_hours = 1.0
    allow_grid_charge = True

    # 1. 데이터 로드 (Data Load)
    loader = SolarDataManager()
    try:
        _, _, test_x, test_y, test_smp = loader.load_and_split_standard(
            os.path.join(BASE_DIR, "data")
        )
    except Exception as e:
        print(f"Error: {e}")
        return

    seq_length = 24
    X_test, y_test = loader.create_sequences(test_x, test_y, seq_length=seq_length)

    # SMP 정렬 (Alignment)
    real_prices = test_smp[seq_length:] if test_smp is not None else None

    print(f"AI 예측 실행 중... (Test: {len(X_test)} hours)")

    # 2. 모델 예측 (Model Predict)
    predictor = LSTMPredictor(os.path.join(BASE_DIR, "src", "lstm_solar_model.pth"))
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
    print(f"Eval -> MAE: {mae:.4f} kW | RMSE: {rmse:.4f} kW | MAPE: {mape:.2f}%")

    # SMP 확인 (Check)
    if real_prices is None or np.sum(real_prices) == 0:
        print("Warning: SMP 데이터가 없어 임시 가격 곡선을 사용합니다.")
        real_prices = [100 if 10 <= i % 24 <= 16 else 200 for i in range(len(y_real_kw))]
    else:
        print(f"Using real SMP (avg: {np.mean(real_prices):.1f})")

    # ---------------------------------------------------------
    # PART 1: 글로벌 배터리 비교 (Benchmark)
    # ---------------------------------------------------------
    print("\n>>> [Part 1] 글로벌 배터리 비교 (Benchmark)")

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
    print(f"0. Baseline (ESS 없음): {int(base_profit):,} KRW")

    avg_price = np.mean(real_prices)

    for batt in batteries:
        profit = 0
        history = []
        for t in range(len(y_real_kw)):
            gen_kw = y_real_kw[t]
            pred_kw = y_pred_kw[t]
            price = real_prices[t]

            action = 0
            if price > avg_price * 1.1:
                action = -1
            elif price < avg_price * 0.9 and pred_kw > 0.1:
                action = 1

            if action == 1:
                charge_request_kw = batt.max_power if allow_grid_charge else min(gen_kw, batt.max_power)
                trade_kw = gen_kw + batt.update(action, charge_request_kw, dt_hours)
            elif action == -1:
                discharge_request_kw = batt.max_power
                trade_kw = gen_kw + batt.update(action, discharge_request_kw, dt_hours)
            else:
                trade_kw = gen_kw

            profit += trade_kw * dt_hours * price
            history.append(profit)

        results[batt.name] = history
        print(f"- {batt.name}: {int(profit):,} KRW")

    ReportGenerator.save_plots(y_real_kw, y_pred_kw, results, baseline_history)

    # ---------------------------------------------------------
    # PART 2: 확장성 테스트 (Scalability)
    # ---------------------------------------------------------
    print("\n>>> [Part 2] 확장성 테스트 (Scalability)")

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

            action = 0
            if price > avg_price * 1.1:
                action = -1
            elif price < avg_price * 0.9 and pred_kw > 0.1:
                action = 1

            if action == 1:
                charge_request_kw = test_batt.max_power if allow_grid_charge else min(gen_kw, test_batt.max_power)
                trade_kw = gen_kw + test_batt.update(action, charge_request_kw, dt_hours)
            elif action == -1:
                discharge_request_kw = test_batt.max_power
                trade_kw = gen_kw + test_batt.update(action, discharge_request_kw, dt_hours)
            else:
                trade_kw = gen_kw

            profit += trade_kw * dt_hours * price
            history.append(profit)

        scalability_results[name] = history
        print(f"  {name}: Final Profit {int(profit):,} KRW")

    ReportGenerator.save_scalability_plot(scalability_results)


if __name__ == "__main__":
    run()

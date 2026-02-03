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
    print("="*60)
    print("💰 SolarX: Real-World Data Simulation")
    print("="*60)

    # 1. 데이터 로드
    loader = SolarDataManager()
    try:
        _, _, test_x, test_y, test_smp = loader.load_and_split_standard(os.path.join(BASE_DIR, 'data'))
    except Exception as e:
        print(f"❌ {e}")
        return

    SEQ_LENGTH = 24
    X_test, y_test = loader.create_sequences(test_x, test_y, seq_length=SEQ_LENGTH)
    
    # SMP 길이 보정
    real_prices = test_smp[SEQ_LENGTH:]
    
    print(f"🔮 AI 모델 예측 중... (Test Set: {len(X_test)} hours)")
    
    # 2. 모델 예측
    predictor = LSTMPredictor(os.path.join(BASE_DIR, 'src', 'lstm_solar_model.pth'))
    y_pred_scaled = predictor.predict(X_test)
    
    y_real_raw = loader.inverse_transform_y(y_test.reshape(-1, 1)).flatten()
    y_pred_raw = loader.inverse_transform_y(y_pred_scaled.reshape(-1, 1)).flatten()
    
    y_real_kw = np.maximum(y_real_raw / 1000.0, 0)
    y_pred_kw = np.maximum(y_pred_raw / 1000.0, 0)
    
    # SMP 데이터 검증
    if real_prices is None or np.sum(real_prices) == 0:
        print("⚠️ 경고: SMP 데이터 없음. 가상 가격 사용.")
        real_prices = [100 if 10 <= i % 24 <= 16 else 200 for i in range(len(y_real_kw))]
    else:
        print(f"📉 실제 SMP 적용 완료! (평균: {np.mean(real_prices):.1f}원)")

    # ---------------------------------------------------------
    # PART 1: 글로벌 배터리 3사 비교 (Benchmark)
    # ---------------------------------------------------------
    print("\n>>> [Part 1] 글로벌 배터리 3사 수익성 비교 시작...")
    
    battery_capacity = np.max(y_real_kw) * 3
    batteries = [
        LGEnergySolution(battery_capacity),
        SamsungSDI(battery_capacity),
        TeslaBattery(battery_capacity)
    ]
    
    results = {}
    baseline_history = []
    
    # 기준(Baseline) 수익
    base_profit = 0
    for t in range(len(y_real_kw)):
        base_profit += y_real_kw[t] * real_prices[t]
        baseline_history.append(base_profit)
    print(f"0. 기존 방식 (No ESS): {int(base_profit):,}원")

    avg_price = np.mean(real_prices)
    
    for batt in batteries:
        profit = 0
        history = []
        for t in range(len(y_real_kw)):
            gen = y_real_kw[t]
            pred = y_pred_kw[t]
            price = real_prices[t]
            
            action = 0
            if price > avg_price * 1.1: action = -1
            elif price < avg_price * 0.9 and pred > 0.1: action = 1
            
            trade = gen + batt.update(action, gen)
            if trade < 0: trade = 0
            profit += trade * price
            history.append(profit)
        
        results[batt.name] = history
        print(f"✅ {batt.name}: {int(profit):,}원")

    ReportGenerator.save_plots(y_real_kw, y_pred_kw, results, baseline_history)

    # ---------------------------------------------------------
    # PART 2: 확장성 테스트 (Scalability)
    # ---------------------------------------------------------
    print("\n>>> [Part 2] 지역별 확장성 검증 (Scalability Test) 시작...")
    
    # 시나리오 정의: (이름, 발전량 계수)
    scenarios = [
        ("Donghae (Base)", 1.0),       # 기준
        ("Jeju (High Solar)", 1.3),    # 발전량 1.3배 (제주도)
        ("Seattle (Low Solar)", 0.6)   # 발전량 0.6배 (흐린 지역)
    ]
    
    scalability_results = {}
    
    for name, factor in scenarios:
        # 시나리오별 발전량 생성
        scenario_gen = y_real_kw * factor
        
        # 챔피언 배터리(Samsung SDI)로 테스트
        test_batt = SamsungSDI(battery_capacity * factor) # 용량도 발전량에 맞춰 스케일업
        
        profit = 0
        history = []
        
        for t in range(len(scenario_gen)):
            gen = scenario_gen[t]
            # 예측값도 비율만큼 변한다고 가정
            pred = y_pred_kw[t] * factor 
            price = real_prices[t] # 가격은 한국 SMP 그대로 적용 (비교를 위해)
            
            action = 0
            if price > avg_price * 1.1: action = -1
            elif price < avg_price * 0.9 and pred > 0.1: action = 1
            
            trade = gen + test_batt.update(action, gen)
            if trade < 0: trade = 0
            
            # 수익 누적
            profit += trade * price
            history.append(profit)
            
        scalability_results[name] = history
        print(f"   📍 {name}: 최종 수익 {int(profit):,}원")

    ReportGenerator.save_scalability_plot(scalability_results)

if __name__ == "__main__":
    run()
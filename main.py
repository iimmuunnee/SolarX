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
    print("💰 [최종] 글로벌 배터리 3사 성적표 (정석 FM 모드)")
    print("="*60)

    # 1. 정석 데이터 로드
    loader = SolarDataManager()
    try:
        # 학습 때랑 똑같이 잘라서 스케일러를 학습시킴 (그래야 정답 복구가 가능)
        _, _, test_x, test_y = loader.load_and_split_standard(os.path.join(BASE_DIR, 'data'))
    except Exception as e:
        print(f"❌ {e}")
        return

    # 2. 테스트 데이터 시퀀스 생성
    SEQ_LENGTH = 24
    X_test, y_test = loader.create_sequences(test_x, test_y, seq_length=SEQ_LENGTH)
    
    print(f"🔮 AI 모델 예측 중... (Test Set: {len(X_test)} hours)")
    
    # 3. 모델 예측
    predictor = LSTMPredictor(os.path.join(BASE_DIR, 'src', 'lstm_solar_model.pth'))
    y_pred_scaled = predictor.predict(X_test)
    
    # 4. 데이터 복구 (나누기 1000 포함)
    y_real_raw = loader.inverse_transform_y(y_test.reshape(-1, 1)).flatten()
    y_pred_raw = loader.inverse_transform_y(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # 단위 변환 (Wh -> kW)
    y_real_kw = y_real_raw / 1000.0
    y_pred_kw = y_pred_raw / 1000.0
    
    y_real_kw = np.maximum(y_real_kw, 0)
    y_pred_kw = np.maximum(y_pred_kw, 0)
    
    # 5. 배터리 시뮬레이션
    battery_capacity = np.max(y_real_kw) * 3
    
    # 가격표
    prices = [100 if 10 <= i % 24 <= 16 else 200 for i in range(len(y_real_kw))]
    
    batteries = [
        LGEnergySolution(battery_capacity),
        SamsungSDI(battery_capacity),
        TeslaBattery(battery_capacity)
    ]
    
    results = {}
    baseline_history = []
    
    # 6. 기준 수익
    base_profit = 0
    for t in range(len(y_real_kw)):
        base_profit += y_real_kw[t] * prices[t]
        baseline_history.append(base_profit)
    
    print(f"0. 기존 방식 (No ESS): {int(base_profit):,}원")

    # 7. 3사 시뮬레이션
    rank = 1
    for batt in batteries:
        profit = 0
        history = []
        for t in range(len(y_real_kw)):
            gen = y_real_kw[t]
            pred = y_pred_kw[t]
            price = prices[t]
            hour = t % 24
            
            action = 0
            # 전략 로직
            if 18 <= hour <= 22: action = -1
            elif 10 <= hour <= 16 and pred > 0.1: action = 1
            
            trade = gen + batt.update(action, gen)
            if trade < 0: trade = 0
            profit += trade * price
            history.append(profit)
        
        results[batt.name] = history
        
        improvement = ((profit - base_profit) / base_profit) * 100 if base_profit != 0 else 0
        print(f"{rank}. {batt.name}: {int(profit):,}원 (+{improvement:.2f}%)")
        rank += 1
        
    # 1. 예측 그래프 (Real vs AI)
    ReportGenerator.plot_prediction(y_real_kw, y_pred_kw)
    
    # 2. 수익 그래프 (전체 & 차액)
    ReportGenerator.plot_benchmark(results, baseline_history)

if __name__ == "__main__":
    run()
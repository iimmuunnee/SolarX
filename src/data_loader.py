import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

class SolarDataManager:
    def __init__(self):
        # 스케일러 정의
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
    def load_and_split_standard(self, data_dir='./data', split_ratio=0.8):
        """
        [정석 모드] Data Leakage 방지 로직
        1. 전체 데이터 로드
        2. 시간순 정렬
        3. 8:2로 분할 (Train / Test)
        4. 오직 'Train' 데이터로만 스케일러 학습(fit)
        5. Test 데이터는 Train의 기준으로 변환(transform)만 수행
        """
        print(f">>> 📂 [FM 모드] '{data_dir}' 데이터 로드 및 정석 분할 중...")
        
        # 1. 파일 로드 및 병합 (기존 로직 동일)
        weather_list = []
        solar_df = None
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"❌ '{data_dir}' 폴더가 없습니다.")

        for filename in os.listdir(data_dir):
            if not filename.endswith('.csv'): continue
            filepath = os.path.join(data_dir, filename)
            try:
                df = pd.read_csv(filepath, encoding='cp949')
            except:
                df = pd.read_csv(filepath, encoding='utf-8')
                
            if '기온(°C)' in df.columns:
                weather_list.append(df)
            elif '01시' in df.columns:
                solar_df = df

        if not weather_list or solar_df is None:
            raise ValueError("❌ 데이터 파일이 부족합니다.")

        # 날씨 병합
        weather_df = pd.concat(weather_list, ignore_index=True)
        if '일시' in weather_df.columns:
            weather_df['Datetime'] = pd.to_datetime(weather_df['일시'])
        weather_df = weather_df.sort_values('Datetime').reset_index(drop=True)

        # 발전량 전처리
        date_col = '날짜' if '날짜' in solar_df.columns else solar_df.columns[0]
        solar_melted = solar_df.melt(id_vars=[date_col],
                                     value_vars=[f'{i:02d}시' for i in range(1, 25)],
                                     var_name='시간_str',
                                     value_name='발전량')
        solar_melted['Hour'] = solar_melted['시간_str'].str.replace('시', '').astype(int) - 1
        solar_melted['Date'] = pd.to_datetime(solar_melted[date_col])
        solar_melted['Datetime'] = solar_melted['Date'] + pd.to_timedelta(solar_melted['Hour'], unit='h')

        # 최종 병합
        req_cols = ['Datetime', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)', '일사(MJ/m2)', '전운량(10분위)']
        for col in req_cols:
            if col not in weather_df.columns: weather_df[col] = 0
        weather_selected = weather_df[req_cols].fillna(0)
        final_data = pd.merge(weather_selected, solar_melted[['Datetime', '발전량']], on='Datetime', how='inner')
        final_data = final_data.sort_values('Datetime').reset_index(drop=True)
        
        # =========================================================
        # 🔥 여기가 정석의 핵심입니다!
        # =========================================================
        # 1. 데이터를 먼저 자릅니다.
        split_idx = int(len(final_data) * split_ratio)
        train_df = final_data.iloc[:split_idx]
        test_df = final_data.iloc[split_idx:]
        
        print(f"✅ 데이터 분할 완료: Train({len(train_df)}개) / Test({len(test_df)}개)")
        
        feature_cols = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)', '일사(MJ/m2)', '전운량(10분위)', '발전량']
        label_col = ['발전량']
        
        # 2. Train 데이터로만 '공부(fit)' 합니다.
        self.scaler_x.fit(train_df[feature_cols])
        self.scaler_y.fit(train_df[label_col])
        
        # 3. 그 기준으로 Train과 Test를 변환(transform) 합니다.
        train_x_scaled = self.scaler_x.transform(train_df[feature_cols])
        train_y_scaled = self.scaler_y.transform(train_df[label_col])
        
        test_x_scaled = self.scaler_x.transform(test_df[feature_cols])
        test_y_scaled = self.scaler_y.transform(test_df[label_col])
        
        return train_x_scaled, train_y_scaled, test_x_scaled, test_y_scaled

    def create_sequences(self, data_x, data_y, seq_length=24):
        """시계열 윈도우 생성"""
        xs, ys = [], []
        for i in range(len(data_x) - seq_length):
            x = data_x[i:i+seq_length]
            y = data_y[i+seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def inverse_transform_y(self, y_scaled):
        return self.scaler_y.inverse_transform(y_scaled)
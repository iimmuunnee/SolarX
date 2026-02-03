import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

class SolarDataManager:
    def __init__(self):
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
    def load_and_split_standard(self, data_dir='./data', split_ratio=0.8):
        print(f">>> 📂 [FM 모드] '{data_dir}' 데이터 통합 로드 중...")
        
        weather_list = []
        solar_df = None
        smp_list = []

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"❌ '{data_dir}' 폴더가 없습니다.")

        # 1. 파일 자동 분류 및 로드
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            filename_lower = filename.lower()
            
            # 지원 확장자 필터
            if not (filename_lower.endswith('.csv') or filename_lower.endswith('.xlsx') or filename_lower.endswith('.xls')):
                continue
            
            # 파일 읽기 (인코딩/형식 자동 대응)
            df = self._read_file_smart(filepath, filename_lower)
            if df is None: continue

            # 컬럼명 공백 제거
            df.columns = df.columns.str.strip()

            # --- [SMP 파일 감지] ---
            # 파일명에 'smp'가 있거나, 컬럼 내용으로 유추
            is_smp = False
            if 'smp' in filename_lower:
                is_smp = True
            elif len(df.columns) > 0 and ('계통한계가격' in str(df.columns[0]) or 'SMP' in str(df.columns[0])):
                is_smp = True
                
            if is_smp:
                print(f"   [가격] SMP 파일 감지: {filename}")
                
                # [핵심] 제목 줄(Header) 처리 로직
                # 만약 '1h'라는 컬럼이 없으면, 첫 줄이 쓸모없는 제목일 확률 99% -> 다시 읽기
                if '1h' not in df.columns:
                    print(f"      ㄴ 헤더 재설정 중... (header=1 적용)")
                    df = self._read_file_smart(filepath, filename_lower, header=1)
                    df.columns = df.columns.str.strip()

                smp_list.append(df)
                
            # --- [날씨 파일 감지] ---
            elif '기온(°C)' in df.columns:
                weather_list.append(df)
                
            # --- [발전량 파일 감지] ---
            elif '01시' in df.columns and ('발전' in filename or 'generation' in filename_lower):
                solar_df = df

        if not weather_list or solar_df is None:
            raise ValueError("❌ 필수 데이터(날씨 또는 발전량)가 부족합니다.")

        # 2. 데이터 병합 및 전처리
        # (1) 날씨
        weather_df = pd.concat(weather_list, ignore_index=True)
        if '일시' in weather_df.columns:
            weather_df['Datetime'] = pd.to_datetime(weather_df['일시'])
        weather_df = weather_df.sort_values('Datetime').reset_index(drop=True)

        # (2) 발전량
        solar_melted = self._melt_data(solar_df, value_name='발전량')

        # (3) SMP (가격) 전처리
        if smp_list:
            smp_df = pd.concat(smp_list, ignore_index=True)
            
            first_col = smp_df.columns[0]
            smp_df.rename(columns={first_col: '날짜'}, inplace=True)
            
            # 날짜 변환 (에러나면 강제로 변환 시도)
            smp_df['날짜'] = pd.to_datetime(smp_df['날짜'].astype(str), format='%Y%m%d', errors='coerce')
            smp_df = smp_df.dropna(subset=['날짜'])
            
            smp_melted = self._melt_smp(smp_df)
        else:
            print("⚠️ SMP 파일을 찾지 못했습니다. (가상 가격 사용 예정)")
            smp_melted = None

        # 3. 최종 병합
        req_cols = ['Datetime', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)', '일사(MJ/m2)', '전운량(10분위)']
        for col in req_cols:
            if col not in weather_df.columns: weather_df[col] = 0
        weather_selected = weather_df[req_cols].fillna(0)

        final_data = pd.merge(weather_selected, solar_melted[['Datetime', '발전량']], on='Datetime', how='inner')
        
        if smp_melted is not None:
            final_data = pd.merge(final_data, smp_melted[['Datetime', 'SMP']], on='Datetime', how='inner')
            print(f"✅ 데이터 병합 완료! (총 {len(final_data)}시간, 실제 SMP 적용)")
        else:
            final_data['SMP'] = 0 

        final_data = final_data.sort_values('Datetime').reset_index(drop=True)

        # 4. Train/Test 분할
        split_idx = int(len(final_data) * split_ratio)
        train_df = final_data.iloc[:split_idx]
        test_df = final_data.iloc[split_idx:]
        
        feature_cols = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)', '일사(MJ/m2)', '전운량(10분위)', '발전량']
        label_col = ['발전량']
        
        self.scaler_x.fit(train_df[feature_cols])
        self.scaler_y.fit(train_df[label_col])
        
        train_x_scaled = self.scaler_x.transform(train_df[feature_cols])
        train_y_scaled = self.scaler_y.transform(train_df[label_col])
        
        test_x_scaled = self.scaler_x.transform(test_df[feature_cols])
        test_y_scaled = self.scaler_y.transform(test_df[label_col])
        
        test_smp = test_df['SMP'].values if 'SMP' in test_df.columns else None
        
        return train_x_scaled, train_y_scaled, test_x_scaled, test_y_scaled, test_smp

    def _read_file_smart(self, filepath, filename_lower, header=0):
        """CSV, Excel, 인코딩 등을 자동으로 처리해서 읽어주는 함수"""
        try:
            if filename_lower.endswith('.csv'):
                try:
                    return pd.read_csv(filepath, encoding='cp949', header=header)
                except:
                    try:
                        return pd.read_csv(filepath, encoding='utf-8', header=header)
                    except:
                         # BOM이 있는 utf-8-sig 시도
                        return pd.read_csv(filepath, encoding='utf-8-sig', header=header)
            else:
                return pd.read_excel(filepath, header=header)
        except Exception as e:
            print(f"⚠️ 파일 읽기 실패 ({filepath}): {e}")
            return None

    def _melt_data(self, df, value_name='Value'):
        date_col = '날짜' if '날짜' in df.columns else df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        hour_cols = [c for c in df.columns if '시' in c and c != date_col]
        melted = df.melt(id_vars=[date_col], value_vars=hour_cols, var_name='시간_str', value_name=value_name)
        melted['Hour'] = melted['시간_str'].str.replace('시', '').astype(int) - 1
        melted['Datetime'] = melted[date_col] + pd.to_timedelta(melted['Hour'], unit='h')
        return melted

    def _melt_smp(self, df):
        if '날짜' in df.columns:
            date_col = '날짜'
        else:
            date_col = df.columns[0]
             
        hour_cols = [f"{i}h" for i in range(1, 25)]
        # 데이터에 있는 컬럼만 골라서 Melt
        available_cols = [c for c in hour_cols if c in df.columns]
        
        melted = df.melt(id_vars=[date_col], value_vars=available_cols, var_name='시간_str', value_name='SMP')
        melted['Hour'] = melted['시간_str'].str.replace('h', '').astype(int) - 1
        melted['Datetime'] = melted[date_col] + pd.to_timedelta(melted['Hour'], unit='h')
        return melted

    def create_sequences(self, data_x, data_y, seq_length=24):
        xs, ys = [], []
        for i in range(len(data_x) - seq_length):
            x = data_x[i:i+seq_length]
            y = data_y[i+seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def inverse_transform_y(self, y_scaled):
        return self.scaler_y.inverse_transform(y_scaled)
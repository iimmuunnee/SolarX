import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


class SolarDataManager:
    def __init__(self):
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def load_and_split_standard(self, data_dir="./data", split_ratio=0.8):
        print(f">>> [FM Mode] '{data_dir}' 데이터 통합 로드 중...")

        weather_list = []
        solar_df = None
        smp_list = []

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"'{data_dir}' 폴더를 찾을 수 없습니다.")

        # 1. 파일 자동 감지 (Auto Detect)
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            filename_lower = filename.lower()

            if not (
                filename_lower.endswith(".csv")
                or filename_lower.endswith(".xlsx")
                or filename_lower.endswith(".xls")
            ):
                continue

            df = self._read_file_smart(filepath, filename_lower)
            if df is None:
                continue

            df.columns = df.columns.str.strip()

            # SMP 파일 감지 (Detect)
            is_smp = False
            if "smp" in filename_lower:
                is_smp = True
            elif len(df.columns) > 0 and (
                "SMP" in str(df.columns[0]) or "system marginal" in str(df.columns[0]).lower()
            ):
                is_smp = True

            if is_smp:
                print(f"   [SMP detected]: {filename}")

                if "1h" not in df.columns:
                    print("      헤더 재시도 (header=1)")
                    df = self._read_file_smart(filepath, filename_lower, header=1)
                    df.columns = df.columns.str.strip()

                smp_list.append(df)

            # 날씨 파일 감지 (Weather)
            elif any("기온" in c for c in df.columns):
                weather_list.append(self._normalize_weather_columns(df))

            # 발전량 파일 감지 (Generation)
            elif any(c.startswith("01") for c in df.columns) and (
                "발전" in filename_lower or "generation" in filename_lower
            ):
                solar_df = df

        if not weather_list or solar_df is None:
            raise ValueError("필수 날씨/발전 데이터가 없습니다.")

        # 2. 병합 (Merge)
        weather_df = pd.concat(weather_list, ignore_index=True)
        if "일시" in weather_df.columns:
            weather_df["Datetime"] = pd.to_datetime(weather_df["일시"])
        weather_df = weather_df.sort_values("Datetime").reset_index(drop=True)

        solar_melted = self._melt_data(solar_df, value_name="발전량")

        if smp_list:
            smp_df = pd.concat(smp_list, ignore_index=True)
            first_col = smp_df.columns[0]
            smp_df.rename(columns={first_col: "날짜"}, inplace=True)
            smp_df["날짜"] = pd.to_datetime(
                smp_df["날짜"].astype(str), format="%Y%m%d", errors="coerce"
            )
            smp_df = smp_df.dropna(subset=["날짜"])
            smp_melted = self._melt_smp(smp_df)
        else:
            print("Warning: SMP 파일이 없어 임시 가격을 사용합니다.")
            smp_melted = None

        req_cols = [
            "Datetime",
            "기온(℃)",
            "강수량(mm)",
            "풍속(m/s)",
            "습도(%)",
            "일조(hr)",
            "일사(MJ/m2)",
            "운량(10분위)",
        ]
        for col in req_cols:
            if col not in weather_df.columns:
                weather_df[col] = 0
        weather_selected = weather_df[req_cols].fillna(0)

        final_data = pd.merge(
            weather_selected, solar_melted[["Datetime", "발전량"]], on="Datetime", how="inner"
        )

        if smp_melted is not None:
            final_data = pd.merge(
                final_data, smp_melted[["Datetime", "SMP"]], on="Datetime", how="inner"
            )
            print(f"Merge complete (총 {len(final_data)}시간, SMP 적용)")
        else:
            final_data["SMP"] = 0

        final_data = final_data.sort_values("Datetime").reset_index(drop=True)

        # 3. Train/Test 분할 (Split)
        split_idx = int(len(final_data) * split_ratio)
        train_df = final_data.iloc[:split_idx]
        test_df = final_data.iloc[split_idx:]

        feature_cols = [
            "기온(℃)",
            "강수량(mm)",
            "풍속(m/s)",
            "습도(%)",
            "일조(hr)",
            "일사(MJ/m2)",
            "운량(10분위)",
            "발전량",
        ]
        label_col = ["발전량"]

        self.scaler_x.fit(train_df[feature_cols])
        self.scaler_y.fit(train_df[label_col])

        train_x_scaled = self.scaler_x.transform(train_df[feature_cols])
        train_y_scaled = self.scaler_y.transform(train_df[label_col])

        test_x_scaled = self.scaler_x.transform(test_df[feature_cols])
        test_y_scaled = self.scaler_y.transform(test_df[label_col])

        test_smp = test_df["SMP"].values if "SMP" in test_df.columns else None

        return train_x_scaled, train_y_scaled, test_x_scaled, test_y_scaled, test_smp

    def _read_file_smart(self, filepath, filename_lower, header=0):
        try:
            if filename_lower.endswith(".csv"):
                try:
                    return pd.read_csv(filepath, encoding="cp949", header=header)
                except Exception:
                    try:
                        return pd.read_csv(filepath, encoding="utf-8", header=header)
                    except Exception:
                        return pd.read_csv(filepath, encoding="utf-8-sig", header=header)
            return pd.read_excel(filepath, header=header)
        except Exception as e:
            print(f"File read failed ({filepath}): {e}")
            return None

    def _normalize_weather_columns(self, df):
        col_map = {}
        for col in df.columns:
            if "기온" in col:
                col_map[col] = "기온(℃)"
            elif "강수량" in col:
                col_map[col] = "강수량(mm)"
            elif "풍속" in col:
                col_map[col] = "풍속(m/s)"
            elif "습도" in col:
                col_map[col] = "습도(%)"
            elif "일조" in col:
                col_map[col] = "일조(hr)"
            elif "일사" in col:
                col_map[col] = "일사(MJ/m2)"
            elif "운량" in col:
                col_map[col] = "운량(10분위)"
        return df.rename(columns=col_map)

    def _melt_data(self, df, value_name="Value"):
        date_col = "날짜" if "날짜" in df.columns else df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        hour_cols = [c for c in df.columns if "시" in c and c != date_col]
        melted = df.melt(
            id_vars=[date_col], value_vars=hour_cols, var_name="시간_str", value_name=value_name
        )
        melted["Hour"] = melted["시간_str"].str.replace("시", "").astype(int) - 1
        melted["Datetime"] = melted[date_col] + pd.to_timedelta(melted["Hour"], unit="h")
        return melted

    def _melt_smp(self, df):
        date_col = "날짜" if "날짜" in df.columns else df.columns[0]
        hour_cols = [f"{i}h" for i in range(1, 25)]
        available_cols = [c for c in hour_cols if c in df.columns]
        melted = df.melt(
            id_vars=[date_col], value_vars=available_cols, var_name="시간_str", value_name="SMP"
        )
        melted["Hour"] = melted["시간_str"].str.replace("h", "").astype(int) - 1
        melted["Datetime"] = melted[date_col] + pd.to_timedelta(melted["Hour"], unit="h")
        return melted

    def create_sequences(self, data_x, data_y, seq_length=24):
        xs, ys = [], []
        for i in range(len(data_x) - seq_length):
            x = data_x[i : i + seq_length]
            y = data_y[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def inverse_transform_y(self, y_scaled):
        return self.scaler_y.inverse_transform(y_scaled)

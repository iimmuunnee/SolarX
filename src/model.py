import torch
import torch.nn as nn
from accelerate import Accelerator

# 코랩 Cell 3번 모델 구조 그대로
class SolarLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, output_size=1, num_layers=1):
        super(SolarLSTM, self).__init__()
        # LSTM 레이어
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully Connected 레이어
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 초기화 (device는 입력 데이터 x를 따라감)
        h0 = torch.zeros(1, x.size(0), 64).to(x.device)
        c0 = torch.zeros(1, x.size(0), 64).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # 마지막 타임스텝만 사용
        out = self.fc(out[:, -1, :])
        return out

class LSTMPredictor:
    def __init__(self, model_path=None):
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        
        # 모델 초기화 (입력 8개, 레이어 1개)
        self.model = SolarLSTM(input_size=8, hidden_size=64, num_layers=1)
        
        if model_path:
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
                print(f"✅ 모델 로드 성공: {model_path}")
            except Exception as e:
                print(f"⚠️ 모델 로드 실패 (랜덤 가중치): {e}")

        self.model = self.accelerator.prepare(self.model)
        self.model.eval()

    def predict(self, input_data):
        if not torch.is_tensor(input_data): # 텐서 형태인지
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
        else:
            input_tensor = input_data
            
        input_tensor = input_tensor.to(self.device) # 데이터를 연산 장치로 이동. 모델과 데이터가 같은 연산 장치에 존재해야 함
        
        with torch.no_grad(): # 메모리 최적화 (학습이 아니라 계산과정을 기록할 이유가 없음)
            pred = self.model(input_tensor)
            
        return pred.cpu().numpy().flatten()
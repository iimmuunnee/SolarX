import torch
import torch.nn as nn
from accelerate import Accelerator


class SolarLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, output_size=1, num_layers=1):
        super(SolarLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class LSTMPredictor:
    def __init__(self, model_path=None):
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        self.model = SolarLSTM(input_size=8, hidden_size=64, num_layers=1)

        if model_path:
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
                print(f"Model loaded: {model_path}")
            except Exception as e:
                print(f"Model load failed (랜덤 가중치 사용): {e}")

        self.model = self.accelerator.prepare(self.model)
        self.model.eval()

    def predict(self, input_data):
        if not torch.is_tensor(input_data):
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
        else:
            input_tensor = input_data

        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            pred = self.model(input_tensor)

        return pred.cpu().numpy().flatten()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
import os
import sys

# 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from src.data_loader import SolarDataManager
from src.model import SolarLSTM

def train():
    print("="*50)
    print("🚀 SolarX 학습")
    print("="*50)
    
    # 1. 정석 데이터 준비
    loader = SolarDataManager()
    
    # 반환값이 5개이므로, 마지막 'test_smp'는 _ 로 받아서 무시
    train_x, train_y, test_x, test_y, _ = loader.load_and_split_standard('./data')
    
    # 시퀀스 생성 (Train만 사용!)
    X_train, y_train = loader.create_sequences(train_x, train_y, seq_length=24)
    
    # 텐서 변환
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    
    print(f"📝 학습 데이터셋 크기: {X_train_t.shape}")
    
    # 2. 모델 설정
    accelerator = Accelerator()
    model = SolarLSTM(input_size=8, hidden_size=64, output_size=1, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
    # 3. 학습 루프
    print("\n>>> 학습 시작 (Epoch 100)...")
    for epoch in range(100):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            accelerator.backward(loss)
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 20 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/100] Loss: {avg_loss:.6f}")
            
    # 4. 저장
    save_path = './src/lstm_solar_model.pth'
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(unwrapped_model.state_dict(), save_path)
    print(f"✅ 모델 저장 완료: {save_path}")

if __name__ == "__main__":
    train()
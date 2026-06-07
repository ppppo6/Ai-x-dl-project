
"""

태양광 발전량 예측 모델 2 - LSTM


"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle

# ============================================================
# 경로 설정
# ============================================================
data_dir   = 'C:/Users/doyun/OneDrive/Desktop/Proj/dataset'
output_dir = 'C:/Users/doyun/OneDrive/Desktop/Proj/models/LSTM/result_lstm'

# ============================================================
# Step 1: 데이터 로드
# ============================================================
print("Step 1: 데이터 로드 중...")
X_train = np.load(f'{data_dir}/X_train.npy')
X_test  = np.load(f'{data_dir}/X_test.npy')
Y_train = np.load(f'{data_dir}/Y_train.npy')
Y_test  = np.load(f'{data_dir}/Y_test.npy')

with open(f'{data_dir}/scaler_Y.pkl', 'rb') as f:
    scaler_Y = pickle.load(f)

print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")

# PyTorch Tensor 변환
X_train_t = torch.FloatTensor(X_train)
Y_train_t = torch.FloatTensor(Y_train).squeeze()
X_test_t  = torch.FloatTensor(X_test)
Y_test_t  = torch.FloatTensor(Y_test).squeeze()

train_dataset = TensorDataset(X_train_t, Y_train_t)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=False)

# ============================================================
# Step 2: LSTM 모델 정의
# ============================================================
print("\nStep 2: LSTM 모델 구축 중...")

class SolarLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2):
        super(SolarLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.1
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = SolarLSTM().to(device)
print(f"디바이스: {device}")
print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}개")

# ============================================================
# Step 3: 학습
# ============================================================
print("\nStep 3: 학습 시작...")

criterion  = nn.MSELoss()
optimizer  = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 50
train_losses = []
test_losses  = []

for epoch in range(EPOCHS):
    # Train
    model.train()
    epoch_loss = 0
    for batch_X, batch_Y in train_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = criterion(pred, batch_Y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Test
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t.to(device))
        test_loss = criterion(test_pred, Y_test_t.to(device)).item()
        test_losses.append(test_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}]  Train Loss: {avg_train_loss:.6f}  Test Loss: {test_loss:.6f}")

# ============================================================
# Step 4: 성능 평가
# ============================================================
print("\nStep 4: 성능 평가 중...")
model.eval()
with torch.no_grad():
    predicted_scaled = model(X_test_t.to(device)).cpu().numpy()

predicted = scaler_Y.inverse_transform(predicted_scaled.reshape(-1, 1))
actual    = scaler_Y.inverse_transform(Y_test.reshape(-1, 1))

mae  = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2   = r2_score(actual, predicted)

print(f"\n{'='*40}")
print(f"  MAE:  {mae:.4f} kW")
print(f"  RMSE: {rmse:.4f} kW")
print(f"  R²:   {r2:.4f}")
print(f"{'='*40}")

# ============================================================
# Step 5: 결과 그래프
# ============================================================
print("\nStep 5: 그래프 생성 중...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Solar Power Prediction - LSTM\nMAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}', fontsize=13)

# 실제 vs 예측
axes[0].plot(actual[:300],    label='Actual',    color='#2196F3', linewidth=1)
axes[0].plot(predicted[:300], label='Predicted', color='#F44336', linewidth=1, linestyle='--')
axes[0].set_title('Actual vs Predicted (First 300)')
axes[0].set_xlabel('Time Step')
axes[0].set_ylabel('Power (kW)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Train/Test Loss
axes[1].plot(train_losses, label='Train Loss', color='#2196F3')
axes[1].plot(test_losses,  label='Test Loss',  color='#F44336')
axes[1].set_title('Loss Curve')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss (MSE)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/lstm_result.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"그래프 저장 완료: {output_dir}/lstm_result.png")

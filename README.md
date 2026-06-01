# 태양광 발전량 예측
AI+X 딥러닝 프로젝트 (G28)

---

Members:

한민채 (경영학과) gksalsc00@hanyang.ac.kr

양희지 (전기공학전공) yangheeji0920@gmail.com

전석호 (정치외교학과) jseokho31@naver.com

김도윤 (데이터사이언스학부) doyun0337@naver.com

---

### 목차

- [I. Proposal](#i-proposal-option-a)
- [II. Datasets](#ii-datasets)
- [III. Methodology](#iii-methodology)
- [IV. Evaluation & Analysis](#iv-evaluation--analysis)
- [V. Related Work](#v-related-work-eg-existing-studies)
- [VI. Conclusion](#vi-conclusion-discussion)

---

## I. Proposal (Option A)

태양광 에너지는 탄소 배출 없이 전력을 생산할 수 있는 핵심적인 재생에너지로, 전 세계적으로 빠르게 보급되고 있습니다. 그러나 태양광 발전량은 일사량, 기온, 습도 등 기상 상황에 따라 변동이 크기 때문에 정확한 예측이 어렵습니다.
태양 일사량을 미리 예측할 수 있다면 다음과 같은 분야에서 활용 가능하다.

전력망 관리: 구름으로 인해 발전량이 갑자기 줄어들 때 백업 전력을 미리 준비 가능
비용 절감: 불필요한 화석연료 사용을 최소화 가능
탄소 배출 감소: 재생에너지를 더 효율적으로 활용 가능

---

## II. Datasets

 사용한 데이터셋) **Solar Radiation Prediction**  (Kaggle에서 찾음)
이 데이터셋은 NASA의 해커톤에서 제공되었으며, 미국 하와이의 HI-SEAS 기상 관측소에 수집된 기상 데이터입니다.  
또한 이 데이터셋은 단일 지점에서 시간 순서대로 기록된 시계열 데이터셋입니다.  


 컬럼에는 UNIXTime, Data, Time, Radiation, Temperature, Pressure, Humidity, WindDirection(Degrees), Speed, TimeSunrise, TimeSunset 이 있으며  
이 중에 저희가 사용할 컬럼들은 Temperature (기온), Humidity (습도), Pressure (기압), Speed (풍속) 입니다. 

---

#### !!위의 데이터셋은 Blog 진행과정을 위해 임시로 넣은 데이터셋입니다. 우선 이 데이터셋은 크게 2개의 문제점을 지닙니다.
#### 첫 번째로 관측기간이 4개월로 매우 짧다는 것입니다. 관측 기간이 짧으면 RNN, LSTM, Transformer 모델들 사이의 차이가 두드러지게 안 날 것입니다. 두 번째로 이 데이터셋은 직접적인 태양광 발전량에 관한 내용이 없다는 것입니다. 대신에 일사량이 있긴 하고 이것이 태양광 발전량과 큰 관련이 있긴 하지만 정확히 발전량을 대변하긴 힘듭니다. (아니면 태양광 발전량과 일사량 사이의 관계식을 따로 찾아 이것까지 고려해볼 수도 있을수도..??)
#### 따라서 현재 다른 많은 데이터셋들을 찾아보고 있는 상태입니다. 저희가 원하는 조건들 (3년치 이상, 기상 상황/태양광 발전량 모두 포함/단일 지역에서 측정/시간이 순서대로 정렬되있어야함 etc..) 을 만족시키는 데이터셋을 찾기 힘들더라고요,,
#### 현재 두 개의 데이터셋을 찾아서 합치는 방법도 고려중입니다.
#### https://re.jrc.ec.europa.eu/pvg_tools/en/ , https://nsrdb.nlr.gov/data-viewer 등 여러 사이트 탐색중입니다!!

---

## III. Methodology

 저희가 사용할 모델은 RNN, LSTM, Transformer 입니다.  
태양 일사량은 시간 순서에 따라 변화하는 시계열 데이터이므로, 순서가 있는 데이터 처리에 특화된 시퀀스 모델 3가지를 선택하여 비교 분석할 예정입니다.

### **RNN** (순환 신경망)  

 RNN은 일반 신경망(MLP)과 달리 이전 출력을 다시 입력에 넣는 순환 구조를 가집니다.  
RNN은 순서가 있는 데이터를 처리하는데 강점을 보입니다. (ex. 시계열 데이터) 또한 RNN의 학습속도가 빠르다는 장점도 있습니다.
하지만 RNN은 치명적인 문제들이 있는데 그 중에 하나는 순서가 긴 데이터에선 앞부분 기억을 까먹는 문제가 있다는 것입니다.  
이것을 기울기 소실 문제 (Vanishing Gradient) 라고 부릅니다.

                                                                                          
### **LSTM** (Long short-term memory)  

 LSTM은 발전된 RNN 기법중 하나에 속합니다.  
기존의 RNN에 Cell State와 Forget Gate, Input Gate, Output Gate를 추가하여 기존 RNN의 기울기 소실 문제를 해결하였습니다.
따라서 LSTM은 장기적인 패턴을 학습할 수 있다는 장점이 존재합니다.

### **Transformer**  

 Transformer는 앞의 2개의 모델에 비해 비교적 최근에 나왔습니다.  
Transformer는 RNN 계열 모델과는 다른 방식을 이용합니다. -> Self Attention
따라서 Transformer는 병렬 처리가 가능하고 데이터가 많을수록 효과적입니다.

{공정한 비교를 하기 위해서 딥러닝 모델을 제외하고 Epoch나 Batch Size, Train/Test 비율 같은 것들은 모두 동일하게 진행할 예정입니다}

{더욱 자세한 설명을 추가할 예정입니다}

---

### 사용할 DL 라이브러리 : PyTorch

{PyTorch에 관한 설명 추가할 예정입니다}

---

## IV. Evaluation & Analysis

```python
print('Work-In-Process')
```
### !!임시 - Blog_In_Progress 제출용!! LLM을 이용해 간단히 코드를 짜서 Transformer를 이용해 그래프 도출해봤습니다 -> 수정 예정
코드:  
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 데이터 로드 및 전처리
# ============================================================
print("=" * 60)
print("Step 1: 데이터 로드 중...")
print("=" * 60)

df = pd.read_csv('solar_prediction.csv')
print(f"데이터 크기: {df.shape}")
print(f"컬럼: {list(df.columns)}")
print(df.head())

# 필요한 컬럼만 선택
# Temperature, Pressure, Humidity, WindDirection, Speed → 입력
# Radiation → 예측 대상 (일사량)
features = ['Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed']
target = 'Radiation'

data = df[features + [target]].copy()
data = data.dropna()  # 결측치 제거

print(f"\n사용할 데이터: {data.shape[0]}행, 입력 변수 {len(features)}개")
print(f"입력 변수: {features}")
print(f"예측 대상: {target}")

# ============================================================
# 2. 정규화 (0~1 사이로 스케일링)
# ============================================================
print("\n" + "=" * 60)
print("Step 2: 데이터 정규화 중...")
print("=" * 60)

scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(data[features].values)
Y_scaled = scaler_Y.fit_transform(data[[target]].values)

print("정규화 완료 (MinMaxScaler: 0~1)")

# ============================================================
# 3. 시계열 시퀀스 만들기
# ============================================================
print("\n" + "=" * 60)
print("Step 3: 시계열 시퀀스 생성 중...")
print("=" * 60)

WINDOW_SIZE = 24  # 과거 24개 데이터를 보고
PRED_SIZE = 1     # 다음 1개 예측

def create_sequences(X, Y, window_size):
    """과거 window_size개 데이터로 다음 값을 예측하는 시퀀스 생성"""
    Xs, Ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i + window_size])
        Ys.append(Y[i + window_size])
    return np.array(Xs), np.array(Ys)

X_seq, Y_seq = create_sequences(X_scaled, Y_scaled, WINDOW_SIZE)
print(f"시퀀스 생성 완료")
print(f"X shape: {X_seq.shape}  → (샘플수, 과거 {WINDOW_SIZE}개, 변수 {len(features)}개)")
print(f"Y shape: {Y_seq.shape}  → (샘플수, 예측값 1개)")

# ============================================================
# 4. Train / Test 분할
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Train/Test 분할 중...")
print("=" * 60)

split = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split], X_seq[split:]
Y_train, Y_test = Y_seq[:split], Y_seq[split:]

print(f"Train: {X_train.shape[0]}개")
print(f"Test:  {X_test.shape[0]}개")

# PyTorch Tensor로 변환
X_train_t = torch.FloatTensor(X_train)
Y_train_t = torch.FloatTensor(Y_train)
X_test_t = torch.FloatTensor(X_test)
Y_test_t = torch.FloatTensor(Y_test)

train_dataset = TensorDataset(X_train_t, Y_train_t)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ============================================================
# 5. Transformer 모델 정의
# ============================================================
print("\n" + "=" * 60)
print("Step 5: Transformer 모델 구축 중...")
print("=" * 60)


class SolarTransformer(nn.Module):
    """
    태양광 일사량 예측을 위한 Transformer 모델

    구조:
    1. Input Projection: 입력 변수(5개)를 d_model 차원으로 확장
    2. Positional Encoding: 시간 순서 정보 부여
    3. Transformer Encoder: Self-Attention으로 패턴 파악
    4. Output Layer: 최종 일사량 예측값 출력
    """

    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(SolarTransformer, self).__init__()

        # 입력을 d_model 차원으로 변환
        self.input_projection = nn.Linear(input_dim, d_model)

        # 위치 인코딩 (순서 정보)
        self.pos_encoding = nn.Parameter(torch.randn(1, WINDOW_SIZE, d_model))

        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 출력 레이어
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch, window_size, input_dim)
        x = self.input_projection(x)       # (batch, 24, 64)
        x = x + self.pos_encoding           # 위치 정보 추가
        x = self.transformer_encoder(x)     # Self-Attention 수행
        x = x[:, -1, :]                     # 마지막 시점만 사용
        x = self.output_layer(x)            # 예측값 출력
        return x


# 모델 생성
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SolarTransformer(input_dim=len(features)).to(device)

# 모델 구조 출력
total_params = sum(p.numel() for p in model.parameters())
print(f"디바이스: {device}")
print(f"모델 파라미터 수: {total_params:,}개")
print(f"\n모델 구조:")
print(model)

# ============================================================
# 6. 모델 학습
# ============================================================
print("\n" + "=" * 60)
print("Step 6: 모델 학습 시작!")
print("=" * 60)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

EPOCHS = 50
train_losses = []
test_losses = []

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    epoch_loss = 0
    for batch_X, batch_Y in train_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

        optimizer.zero_grad()
        prediction = model(batch_X)
        loss = criterion(prediction, batch_Y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # --- Test ---
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t.to(device))
        test_loss = criterion(test_pred, Y_test_t.to(device)).item()
        test_losses.append(test_loss)

    scheduler.step(test_loss)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}]  "
              f"Train Loss: {avg_train_loss:.6f}  "
              f"Test Loss: {test_loss:.6f}")

print("\n학습 완료!")

# ============================================================
# 7. 예측 및 성능 평가
# ============================================================
print("\n" + "=" * 60)
print("Step 7: 성능 평가")
print("=" * 60)

model.eval()
with torch.no_grad():
    predicted_scaled = model(X_test_t.to(device)).cpu().numpy()

# 원래 스케일로 복원
predicted = scaler_Y.inverse_transform(predicted_scaled)
actual = scaler_Y.inverse_transform(Y_test)

# 성능 지표 계산
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)

print(f"\n{'='*40}")
print(f"  MAE  (평균 절대 오차):  {mae:.2f} W/m²")
print(f"  RMSE (제곱근 평균 오차): {rmse:.2f} W/m²")
print(f"  R²   (결정 계수):      {r2:.4f}")
print(f"{'='*40}")

if r2 > 0.9:
    print("  → 우수한 예측 성능!")
elif r2 > 0.7:
    print("  → 양호한 예측 성능")
else:
    print("  → 개선이 필요한 수준 (하이퍼파라미터 튜닝 권장)")

# ============================================================
# 8. 시각화 (그래프 4개)
# ============================================================
print("\n" + "=" * 60)
print("Step 8: 결과 시각화 중...")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Solar Radiation Prediction - Transformer Model', fontsize=16, fontweight='bold')

# --- 그래프 1: 실제 vs 예측 (전체) ---
ax1 = axes[0, 0]
ax1.plot(actual[:300], label='Actual', color='#2196F3', alpha=0.8, linewidth=1)
ax1.plot(predicted[:300], label='Predicted', color='#F44336', alpha=0.8, linewidth=1, linestyle='--')
ax1.set_title('Actual vs Predicted (First 300 samples)', fontsize=12)
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Solar Radiation (W/m²)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- 그래프 2: 산점도 ---
ax2 = axes[0, 1]
ax2.scatter(actual, predicted, alpha=0.3, s=10, color='#4CAF50')
max_val = max(actual.max(), predicted.max())
ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax2.set_title(f'Scatter Plot (R² = {r2:.4f})', fontsize=12)
ax2.set_xlabel('Actual Radiation (W/m²)')
ax2.set_ylabel('Predicted Radiation (W/m²)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- 그래프 3: 학습 곡선 ---
ax3 = axes[1, 0]
ax3.plot(train_losses, label='Train Loss', color='#2196F3')
ax3.plot(test_losses, label='Test Loss', color='#F44336')
ax3.set_title('Training & Test Loss Curve', fontsize=12)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss (MSE)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# --- 그래프 4: 오차 분포 ---
ax4 = axes[1, 1]
errors = (actual - predicted).flatten()
ax4.hist(errors, bins=50, color='#FF9800', alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax4.set_title(f'Prediction Error Distribution (MAE={mae:.2f})', fontsize=12)
ax4.set_xlabel('Error (W/m²)')
ax4.set_ylabel('Frequency')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('transformer_result.png', dpi=150, bbox_inches='tight')
plt.show()
print("그래프 저장 완료: transformer_result.png")

# ============================================================
# 9. 결과 요약
# ============================================================
print("\n" + "=" * 60)
print("최종 결과 요약")
print("=" * 60)
print(f"""
모델: Transformer (Encoder)
  - d_model: 64
  - heads: 4
  - layers: 2
  - 총 파라미터: {total_params:,}개

데이터:
  - 입력: {features}
  - 출력: Solar Radiation (W/m²)
  - Window: 과거 {WINDOW_SIZE}개 → 다음 1개 예측
  - Train/Test: {X_train.shape[0]} / {X_test.shape[0]}

성능:
  - MAE:  {mae:.2f} W/m²
  - RMSE: {rmse:.2f} W/m²
  - R²:   {r2:.4f}
""")
```

<img width="1500" height="1100" alt="image" src="https://github.com/user-attachments/assets/814eed5a-ff2f-4bc5-bfdf-1e6111902568" />
(걸린시간 : 약 20분)

{평가 지표 : MAE, RMSE, R^2 ??}

---

## V. Related Work (e.g., existing studies)

사용한 데이터셋 :  

(https://www.kaggle.com/datasets/dronio/SolarEnergy)

Transformer 관련 내용 :  강의자료 07-beyond

---

## VI. Conclusion: Discussion

{3개 모델 중 어느 것이 가장 정확했는지 확인할 예정입니다}
{또한 각 모델마다 두드러지게 나타나는 특징들을 확인할 예정입니다}

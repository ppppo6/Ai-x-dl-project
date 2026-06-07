
"""

최종 전처리
정규화 / 시퀀스 생성 / Train-Test 분할

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# 경로 설정
# ============================================================
input_file = 'C:/Users/doyun/OneDrive/Desktop/Proj/data_preprocessing/merge_dataset/merged_dataset.csv'
output_dir = 'C:/Users/doyun/OneDrive/Desktop/Proj/dataset'

# ============================================================
# Step 1: 데이터 로드
# ============================================================
print("Step 1: 데이터 로드 중...")
df = pd.read_csv(input_file)
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"데이터: {len(df):,}행")
print(df.head())

# ============================================================
# Step 2: 입력 변수 / 예측 대상 분리
# ============================================================
print("\nStep 2: 입력/출력 분리 중...")
feature_cols = ['Temperature', 'Clearsky GHI', 'GHI',
                'Relative Humidity', 'Pressure', 'Wind Speed']
target_col   = 'power_kW'

X = df[feature_cols].values  # 입력 (기상 변수 6개)
Y = df[target_col].values    # 출력 (발전량)

print(f"입력 변수: {feature_cols}")
print(f"예측 대상: {target_col}")

# ============================================================
# Step 3: 정규화 (0~1)
# ============================================================
print("\nStep 3: 정규화 중...")
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1))

print("MinMaxScaler 정규화 완료 (0~1)")

# ============================================================
# Step 4: 시퀀스 생성
# ============================================================
print("\nStep 4: 시퀀스 생성 중...")
WINDOW_SIZE = 168  # 30분 × 48 = 24시간

def create_sequences(X, Y, window):
    Xs, Ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window])   # 과거 48개 기상 데이터
        Ys.append(Y[i+window])     # 다음 시점 발전량
    return np.array(Xs), np.array(Ys)

X_seq, Y_seq = create_sequences(X_scaled, Y_scaled, WINDOW_SIZE)
print(f"시퀀스 생성 완료")
print(f"X shape: {X_seq.shape}  → (샘플수, {WINDOW_SIZE}개, {len(feature_cols)}개 변수)")
print(f"Y shape: {Y_seq.shape}  → (샘플수, 1)")

# ============================================================
# Step 5: Train / Test 분할 (80% / 20%)
# ============================================================
print("\nStep 5: Train/Test 분할 중...")
split = int(len(X_seq) * 0.8)

X_train, X_test = X_seq[:split], X_seq[split:]
Y_train, Y_test = Y_seq[:split], Y_seq[split:]

print(f"Train: {len(X_train):,}개 ({len(X_train)/len(X_seq)*100:.0f}%)")
print(f"Test:  {len(X_test):,}개  ({len(X_test)/len(X_seq)*100:.0f}%)")

# ============================================================
# Step 6: 저장
# ============================================================
import os
os.makedirs(output_dir, exist_ok=True)

np.save(f'{output_dir}/X_train.npy', X_train)
np.save(f'{output_dir}/X_test.npy',  X_test)
np.save(f'{output_dir}/Y_train.npy', Y_train)
np.save(f'{output_dir}/Y_test.npy',  Y_test)

# scaler도 저장 (나중에 역정규화에 필요)
import pickle
with open(f'{output_dir}/scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open(f'{output_dir}/scaler_Y.pkl', 'wb') as f:
    pickle.dump(scaler_Y, f)

print("\n" + "="*50)
print("✅ 최종 전처리 완료!")
print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"Y_train: {Y_train.shape}")
print(f"Y_test:  {Y_test.shape}")
print(f"저장 위치: {output_dir}")
print("="*50)

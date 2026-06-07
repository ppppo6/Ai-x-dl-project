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
- [III. Data Preprocessing](#iii-data-preprocessing)
- [IV. Methodology](#iv-methodology)
- [V. Evaluation & Analysis](#v-evaluation--analysis)
- [VI. Related Work](#vi-related-work-eg-existing-studies)
- [VII. Conclusion](#vii-conclusion-discussion)

---

# I. Proposal (Option A)

### 프로젝트 개요

최근 탄소중립과 친환경 에너지에 대한 관심이 증가하면서 태양광 발전의 활용 범위가 빠르게 확대되고 있습니다. 
 
그러나 태양광 발전은 일사량, 기온, 습도, 풍속 등 다양한 기상 상황에 영향을 받기 때문에 발전량이 일정하지 않다는 한계가 있습니다. 따라서 전력을 안정적이게 수급하고 비용을 관리하기 위해서는 태양광 발전량을 예측하는 기술이 중요합니다.

### 프로젝트 목표

본 프로젝트의 목표는 실제 태양광 발전량 데이터와 기상 데이터를 활용하여 태양광 발전량을 예측하는 딥러닝 모델을 구축해서 비교하는 것입니다.

동일한 조건 아래 RNN, LSTM, Transformer 이 3개의 시퀀스 모델을 통해 태양광 발전량을 예측하고, 각 모델의 성능을 MAE, RMSE, R²를 기준으로 비교 분석할 예정입니다.

여기서 시퀀스 모델이란 순서가 있는 데이터 처리에 특화된 딥러닝 모델로, 시간 순서에 따라 변화하는 태양광 발전량 예측에 적합합다고 생각합니다.

# II. Datasets

 이 프로젝트를 위해서 저희가 사용한 데이터셋은 2개입니다.

태양광 발전량을 예측할려면 **실제 발전소**에서의 발전량이랑 그 발전소 주위의 **기상 상황**에 대한 정보가 필요합니다. 

하지만 '단일 발전소에서의 발전량 / 그 주위의 기상 상황 / 최소 1년치 이상' 을 한번에 만족시키는 데이터셋을 찾기 어려웠습니다.

따라서 동일한 조건 아래 입력 변수와 예측 대상에 대한 데이터셋을 따로 구해서 합치는 작업을 진행했습니다.

---

 ## 사용한 데이터셋 1) NIST - Photovoltaic Data

### 데이터셋 설명

 이 데이터셋은 미국 메릴랜드주 게이더스버그에 위치한 미국 국립표준기술연구소(NIST)에 설치된 태양광 어레이에서 수집된 데이터입니다. 여기서 태양광 어레이란 태양광 패널 여러개를 직렬 또는 병렬로 연결한 것입니다.

이 데이터셋의 측정 간격은 1분이며, NIST 사이트에는 4곳에서 측정한 데이터가 존재해서 저희가 원하는 1곳을 골라서 따로 데이터를 얻을 수 있습니다. (Roof로 선택함)

저희가 얻은 데이터는 2017년 1월 1일부터 2017년 12월 31까지의 데이터입니다.

또한 이 데이터셋은 무료로 오픈되어있어 누구나 사용할 수 있습니다. 

<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/22f36a87-af28-40e2-94a9-c47ef0cd93c4" />

  (태양광 발전소 사진)

  --

### 데이터셋 특징

원래 이 데이터셋에는 발전량 이외에도 많은 변수들(ex. 장비 상태)이 있습니다.

하지만 저희가 여기서 필요했던 것은 발전량이였기 때문에 1개의 변수만 사용했습니다.

| 변수 | 변수명 | 설명 |
| --- | --- | --- |
| 발전량 | InvPAC_kW_Avg | 인버터를 통해 변환된 실제 교류(AC) 출력 전력의 평균값이다. 1분마다 측정되었으며 단위는 kW이다. |

여기서 인버터란 태양광 패널이 생산한 직류(DC) 전기를 실생활에서 사용 가능한 교류(AC)로 변환하는 장치입니다.

---

 ## 사용한 데이터셋 2) NSRDB - Weather Data

### 데이터셋(사이트) 설명

 이 데이터셋은 미국 국립재생에너지연구소(NREL)에서 공개한 태양 복사량 및 기상 상황 데이터베이스입니다. (NSRDB: National Solar Radiation Database)

NSRDB 사이트의 특징 중 하나는 자신의 원하는 지역의 데이터를 자기가 원하는 변수만 골라서 데이터를 얻을 수 있다는 점입니다. (년도도 선택 가능)

 이를 이용해 저희는 태양광 발전량이 측정된 장소 (미국 게이더스버그)에서의 2017년 데이터를 얻을 수 있었습니다. 또한 여러 변수들 중에서 태양광 발전량에 크게 영향을 미친다고 판단한 변수 6개를 골라서 데이터셋을 최종적으로 얻었습니다.

 --

 ### 데이터셋 특징

| 변수 | 변수명 | 설명 |
| --- | --- | --- |
| 기온 | Temperature | 그 지역의 기온을 나타내며, 기온이 높으면 패널 효율이 떨어져 발전량이 낮아질 수도 있다. <br> (단위 : °C) |
| 맑은 하늘 기준 일사량 | Clearsky GHI | 실제 일사량(GHI)과 비교해 구름 영향 파악하기 위해 쓰였다. <br> (단위 : W/m²) |
| 실제 일사량 | GHI | 실제 지면에 닿는 태양 복사량을 뜻하며 발전량과 가장 직접적으로 비례한다. <br> (단위 : W/m²) |
| 상대 습도 | Relative Humidity | 상대 습도가 높을수록 구름or안개 가능성이 높아 발전량 감소 가능성도 높아진다. <br> (단위 : %) |
| 기압 | Pressure | 기압 패턴으로 날씨 변화 예측이 가능하다. <br> (단위 : mbar) |
| 풍속 | Wind Speed | 패널 냉각 효과로 인해 발전 효율에 영향을 끼칠거라고 생각하였다. <br> (단위 : m/s) |

---

 #### 두 데이터셋의 공통점

 1. 측정된 장소 (미국 메릴랜드주 게이더스버그)
 2. 측정된 시간 (2017/01/01 ~ 2017/12/31)
 3. 시간 기준 (미국 동부 표준시 / UTC-5)

---

# III. Data Preprocessing

 ## 1. 발전량 데이터 (PV.Generation) 전처리

 ### 1-1. 파일 병합시키기

  원본 데이터파일인 '2017' 파일은 원래 1년(365일)치의 데이터가 한번에 모아져 있는게 아니였습니다. 2017 폴더 안에는 달에 맞춰 01부터 12까지 12개의 하위 폴더들이 있었고, 또 그 안에는 날짜별 csv 파일이 있었습니다.

따라서 우선적으로 이 365개의 csv파일들을 하나로 합치는 작업을 수행했습니다.

 또한 파일을 하나로 합칠 때, 실제 사용할 발전량 컬럼을 제외한 나머지 컬럼들은 삭제했습니다.

+) 변수명 InvPAC_kW_Avg을 power_kW 로 변경했습니다.

[preprocessing 1-1.py](./preprocessing/preprocessing_1-1.py)

 ```python

"""코드 일부분"""

# 합치기
df_list = []
for i, file in enumerate(all_files):
    try:
        df = pd.read_csv(file)
        df = df[['TIMESTAMP', 'InvPAC_kW_Avg']].copy()
        df_list.append(df)
        if (i+1) % 50 == 0:
            print(f"{i+1}/{len(all_files)} 완료...")
    except Exception as e:
        print(f"⚠️ {os.path.basename(file)} 오류: {e}")

# 저장
combined = pd.concat(df_list, ignore_index=True)
combined.columns = ['timestamp', 'power_kW']
combined = combined.sort_values('timestamp').reset_index(drop=True)
combined.to_csv(output_file, index=False)
```

 ### 1-2. 전처리①

 이전에는 365개의 파일을 1개의 csv 파일로 합쳤었습니다.

 이번에 수행한 과정은 총 5개입니다.

 **1. 데이터에 불필요한 타임존을 제거하기 (UTC-5)**

발전량 데이터랑 기상 상황 데이터 둘다 시간이 미국 동부 표준시 (UTC-5) 로 이미 동일하게 맞춰져 있었습니다. 따라서 데이터에 있는 타임존을 필요없다 판단해 삭제했습니다.

**2. 30분 단위로 리샘플링하기**

기존의 1분 단위로 측정된 데이터는 한번에 다루기에 양이 너무 방대합니다. 또한 기상 상황 데이터가 30분 단위로 측정된 것이여서 발전량 데이터를 1분에서 30분으로 리샘플링하는 과정을 거쳤습니다.

**3. 발전량 값이 음수로 나온 경우를 모두 0으로 처리하기**

원래는 밤에는 햇빛이 없기 때문에 발전량이 0으로 나오는게 정상입니다. 하지만 센서 노이즈같은 이유때문에 아주 미세한 전류 흐름이나 노이즈가 측정되어 발전량이 음수로 나오는 경우가 있습니다. 발전량이 음수인 것은 물리적으로 불가능하기 때문에 음수값을 모두 0으로 처리했습니다.

**4. 결측치 제거하기**

결측치가 존재할 경우 모델 학습에 오류가 날 수도 있기에 제거해줬습니다. (1개 발견)

**5. 발전량을 소수점 반올림하기 (3자리)**

발전량이 0.6677777 같이 나오는 경우 소수점 3~4자리 이후부터는 거의 무의미하기 때문에 깔끔한 데이터 처리를 위해 반올림했습니다.

[preprocessing 1-2.py](./preprocessing/preprocessing_1-2.py)

```python

"""코드 일부분"""

# ============================================================
# Step 3: 30분 단위 리샘플링
# ============================================================
print("\nStep 3: 30분 단위 리샘플링 중...")
df = df.set_index('timestamp')

# 30분 평균
df_30min = df['power_kW'].resample('30T').mean()
df_30min = df_30min.reset_index()
df_30min.columns = ['timestamp', 'power_kW']

print(f"리샘플링 완료: {len(df_30min):,}행 (30분 단위)")

```

 ## 2. 기상 상황 데이터 (Weather_Data) 전처리

  ### 2-1. 전처리②

 이번에도 총 5개의 과정들을 수행하였습니다.

**1. 첫 2줄 제거하기**

기존 데이터의 첫 2줄은 데이터에 대한 설명이였습니다. 발전량 데이터와 동일한 형식을 갖추기 위해서 제거해줬습니다.

**2. 시간 컬럼 생성하기**

원본 데이터에는 컬럼이 년도/달/날짜/시간/분 다섯개로 쪼개져있었습니다. 이것도 발전량 데이터와 동일한 형식(timestamp)을 갖추기 위해서 timestamp 컬럼을 생성하고 기존 5개 컬럼을 제거하였습니다.

**3. 결측치 제거하기**

1-2와 동일하게 결측치가 존재할 경우 모델 학습에 오류가 날 수도 있기에 제거해줬습니다. (0개 발견)

**4. 소수점 반올림하기 (3자리)**

1-2과정과 동일하게 소수점 3~4자리 이후부터는 거의 무의미하기 때문에 깔끔한 데이터 처리를 위해 반올림했습니다.

**5. 일사량이 음수로 나온경우 0으로 처리하기**

발전량 데이터를 전처리 할때 발전량이 음수가 나온 경우가 있어서 혹시나 하는 마음에 이 코드도 넣었습니다.

[preprocessing 2-1.py](./preprocessing/preprocessing_2-1.py)

 ```python

"""코드 일부분"""

# ============================================================
# Step 2: 시간 컬럼 생성
# ============================================================
print("\nStep 2: 시간 컬럼 생성 중...")
df['timestamp'] = pd.to_datetime(
    df['Year'].astype(str) + '-' +
    df['Month'].astype(str).str.zfill(2) + '-' +
    df['Day'].astype(str).str.zfill(2) + ' ' +
    df['Hour'].astype(str).str.zfill(2) + ':' +
    df['Minute'].astype(str).str.zfill(2)
)
print(f"시간 컬럼 생성 완료: {df['timestamp'].iloc[0]}")

# 기존 시간 컬럼 제거
df = df.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'])
print("기존 시간 컬럼 제거 완료")

```

 ## 3. 두 데이터셋 합치기

 여기서는 **NIST 발전량 데이터셋**과 **NSRDB 기상 상황 데이터**를 **timestamp** 컬럼 기준으로 합쳤습니다.

 이때 발전량 데이터는 결측치 제거를 할 때 1행이 사라져서 기상 상황 데이터보다 1행이 더 적은 상태였습니다. 하지만 둘을 합칠 때 Inner Join 형식으로 해서 문제가 생기지는 않았습니다.

  (Inner Join 형식을 사용하면 두 데이터셋에 모두 존재하는 timestamp만 합칠 수 있다. 따라서 문제가 안생김)

 [preprocessing_3.py](./preprocessing/preprocessing_3.py)

 ```python

"""코드 일부분"""

# ============================================================
# Step 3: timestamp 기준으로 합치기
# ============================================================
print("\nStep 3: 데이터 합치는 중...")
df = pd.merge(power, weather, on='timestamp', how='inner')
print(f"합치기 완료: {len(df):,}행")

```

 ## 4. 최종 전처리 과정

마지막으로 합친 데이터셋을 최종적으로 전처리하는 과정을 진행하였습니다. 어찌보면 전처리 과정에서 가장 중요한 작업이라고 할 수도 있겠습니다.

**1. 컬럼들을 입력 변수 / 예측 대상 으로 분리하기**

입력 변수 : Temperature , Clearsky GHI , GHI , Relative Humidity , Pressure , Wind Speed (기상 상황)  /  예측 대상 : power_kW (발전량)

**2. 정규화하기**

변수마다 숫자의 범위가 달라 정규화를 하지 않으면 모델이 특정 변수에 편향될 수도 있습니다. 이를 방지하기 위해 MinMaxScaler를 이용해 정규화를 해 변수들의 범위를 0~1로 바꾸었습니다.

+) 역정규화 과정도 추가했습니다. (역정규화 : 예측 결과를 실제 단위(kW)로 해석하기 위해 정규화 이전 값으로 되돌리는 과정)

**3. 시퀀스 생성하기 (Window Size = 168)**

과거 168개 데이터(3.5일치)를 하나의 묶음으로 만들어 다음 시점(30분)의 발전량을 예측하는 형태로 변환하였습니다.

+) 원래 Window Size를 48로 하려고 했으나 RNN, LSTM, Transformer 사이의 성능 차이가 크게 나지 않을 것을 우려하여 168로 늘렸습니다.
 <br> (48 - 1일치 / 168 - 3.5일치)

**4. Train / Test 분할하기 (Train - 80% / Test - 20%)**

시계열 데이터의 특성상 시간 순서대로 데이터셋의 앞 80%를 Train, 뒤 20%를 Test로 분할해야 했습니다.

 [preprocessing_4.py](./preprocessing/preprocessing_4.py)

```python

"""코드 일부분"""

# ============================================================
# Step 4: 시퀀스 생성
# ============================================================
print("\nStep 4: 시퀀스 생성 중...")
WINDOW_SIZE = 168  #3.5일치

def create_sequences(X, Y, window):
    Xs, Ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window])   # 과거 168개 기상 데이터
        Ys.append(Y[i+window])     # 다음 시점 발전량
    return np.array(Xs), np.array(Ys)

X_seq, Y_seq = create_sequences(X_scaled, Y_scaled, WINDOW_SIZE)
print(f"시퀀스 생성 완료")
print(f"X shape: {X_seq.shape}  → (샘플수, {WINDOW_SIZE}개, {len(feature_cols)}개 변수)")
print(f"Y shape: {Y_seq.shape}  → (샘플수, 1)")

```
---

[데이터셋 다운로드](https://www.kaggle.com/datasets/ppppo6/dataset-proj)

각 과정에서 나온 결과 파일과 최종으로 나온 파일들을 Kaggle에 업로드하였습니다. Github에 올리려 했지만 용량이 너무 커서 올리지 못했습니다.

---

# IV. Methodology

모델을 구축하기 앞서 최종 전처리 과정을 통해 만들어진 파일에 대해서 간단히 알아보겠습니다.

| 파일명 | 설명 |
| ---| --- |
| **X_train.npy** | 13880개의 샘플이 있으며, 각 샘플은 과거 168개 시점의 기상 변수 6개로 구성되어있다. |
| **X_test.npy** | 3471개의 샘플이 있으며, 각 샘플은 과거 168개 시점의 기상 변수 6개로 구성되어있다. <br> 이것은 모델이 테스트를 거칠 때 처음 보는 데이터로 성능 평가에 사용되기도 한다. |
| **Y_train.npy** | 13880개의 샘플이 있으며, 각 샘플은 X_train의 각 샘플에 대응하는 실제 발전량 값이다. |
| **Y_test.npy** | 3471개의 샘플이 있으며, 각 샘플은 X_test의 각 샘플에 대응하는 실제 발전량 값이다. |
| **scaler_X.pkl** | 입력 변수 6개를 정규화할 때 사용한 MinMaxScaler 저장 파일이다. |
| **scaler_Y.pkl** | 발전량을 정규화할 때 사용한 MinMaxScaler 저장 파일이다. <br> 모델 예측값(0~1)을 실제 발전량(kW)으로 역정규화할 때 사용하기도 한다. |

---

또한 공정한 비교를 위해서 모델을 제외한 나머지 학습 조건같은 것들을 모두 동일하게 맞추고 시작했습니다.

### 동일하게 맞춘 것들

#### 데이터

| 항목 | 설정값 |
|------|--------|
| 데이터셋 | X_train.npy, X_test.npy, Y_train.npy, Y_test.npy |
| 입력 변수 | Temperature, Clearsky GHI, GHI, Relative Humidity, Pressure, Wind Speed |
| 예측 대상 | power_kW |
| 정규화 | MinMaxScaler (0~1) |
| Window Size | 168 |
| Train/Test 비율 | 80% / 20% |

#### 학습 조건

| 항목 | 설정값 |
|------|--------|
| Epoch | 50 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | MSELoss |

#### 모델 내부 구조

| 항목 | 설정값 |
|------|--------|
| hidden_size | 64 |
| num_layers | 2 |
| dropout | 0.1 |
| 출력층 | Linear(64→32) → ReLU → Linear(32→1) |

#### 평가 지표

| 항목 | 설정값 |
|------|--------|
| 평가 지표 | MAE, RMSE, R² |
| 평가 데이터 | X_test, Y_test |

---

마지막으로 시작하기 전에 PyTorch에 대해서 간단하게 알아보겠습니다.

PyTorch란

---

## 1. RNN 모델 구축

### 1-1. RNN 모델 설명

### 1-2. RNN 모델 코딩

#### STEP 0. 필요한 패키지

```python

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset  ##torch에서 DataLoader, TensorDataset을 가져옴.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  ##scikit-learn에서 MAE, RMSE, R² 계산 함수를 불러옴.
import matplotlib.pyplot as plt
import pickle

```

RNN 모델 구축에 필요한 패키지들은 torch, numpy, scikit-learn, matplotlib, pickle이 있습니다.

| 패키지 | 용도 |
|--------|------|
| torch | RNN/LSTM/Transformer 모델을 구축하고 학습할 때 사용됨 |
| numpy | .npy 파일을 로드하고 수치 연산을 할 때 사용됨 |
| scikit-learn | MinMaxScaler(정규화)나 MAE, RMSE, R² 값을 구할 때 사용됨  |
| matplotlib | 그래프 시각화 작업을 할 때 사용됨 |
| pickle | .pkl 파일을 로드할 때 사용됨 (Python 기본 내장 패키지) |

+) torch의 하위 패키지인 DataLoader, TensorDataset는 데이터를 묶을 수 있게 해주어 많은 데이터를 나눠서 처리할 수 있게 해줍니다.

---

#### STEP 1. 데이터 로드하기

```python

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

```

**코드 설명**

```python

X_train = np.load(f'{data_dir}/X_train.npy')
X_test  = np.load(f'{data_dir}/X_test.npy')
Y_train = np.load(f'{data_dir}/Y_train.npy')
Y_test  = np.load(f'{data_dir}/Y_test.npy')

```

최종 데이터 전처리 과정에서 만들었던 4개의 .npy 파일을 로드했습니다.

```python

with open(f'{data_dir}/scaler_Y.pkl', 'rb') as f:
    scaler_Y = pickle.load(f)

```

최종 데이터 전처리 과정에서 만들었던 scaler_Y.pkl 파일을 로드했습니다.

이것은 나중에 모델 예측값(0~1)을 실제 발전량(kW)으로 역정규화할 때 사용됩니다.

```python

X_train_t = torch.FloatTensor(X_train)
Y_train_t = torch.FloatTensor(Y_train).squeeze()
X_test_t  = torch.FloatTensor(X_test)
Y_test_t  = torch.FloatTensor(Y_test).squeeze()

```

PyTorch 모델은 Tensor 형태로만 입력을 받을 수 있기 때문에 numpy 배열을 PyTorch Tensor형태로 데이터 형식을 변환했습니다.

또한 `.squeeze()`의 기능은 Y의 크기(shape)의 차원을 축소시키는 것입니다. 차원을 축소시키는 이유는 모델 출력값과 shape를 
맞추기 위함입니다.

```python

train_dataset = TensorDataset(X_train_t, Y_train_t)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=False)

```

`TensorDataset`은 X_train과 Y_train을 하나의 쌍으로 묶어줍니다. 또한 `DataLoader`는 묶인 데이터를 64개씩 나눠서 모델에 전달해주는 역할을 합니다.

또한 저희가 다루는 데이터는 시계열 데이터여서 순서를 섞으면 안됩니다. 따라서 `shuffle=False`로 했습니다.

---

#### Step 2. RNN 모델 구축하기

```python

# ============================================================
# Step 2: RNN 모델 정의
# ============================================================
print("\nStep 2: RNN 모델 구축 중...")

class SolarRNN(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2):
        super(SolarRNN, self).__init__()
        self.rnn = nn.RNN(
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
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = SolarRNN().to(device)
print(f"디바이스: {device}")
print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}개")

```

**코드 설명**

```python

class SolarRNN(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2):
        super(SolarRNN, self).__init__()

```

SolarRNN이라는 모델 클래스를 생성했습니다.

`hidden_size=64` : 기억 저장 공간 뉴런 수가 64개입니다.

`num_layers=2` : RNN 층을 2개 쌓았습니다.

`super()` 의 기능은 nn.Module의 기능을 이어받아 사용할 수 있게 초기화해주는 것입니다.

```python

 self.rnn = nn.RNN(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.1
        )

```

여기선 RNN 모델을 정의했습니다.

`batch_first = True` : batch 데이터를 맨 처음으로 해줘 데이터 순서를 (배치, 시간, 변수) 순서로 설정하게 해줍니다. --> (데이터 순서 통일하기)

`dropout = 0.1` : 학습 시에 뉴런의 10%를 무작위로 꺼서 과적합이 일어나는 것을 방지해줍니다. 여기서 과적합이란 모델이 학습 데이터에 너무 의존을 해 새로운 데이터를 잘 못맞추는 상태를 뜻합니다.

```python

 self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

```

여기선 FC층을 정의했습니다. FC층이란 모델의 마지막 부분에서 최종 예측값을 출력하는 층이며 완전연결층이라고 부르기도 합니다.

`nn.Linear(hidden_size, 32)` : 64차원 (hidden_size=64)을 32차원으로 축소해줍니다.

`nn.ReLU()` : 복잡한 비선형 패턴 학습을 가능하게 해주는 활성화 함수이며 중간층에 적용되었습니다.

` nn.Linear(32, 1)` : 32차원을 발전량 숫자 1개로 출력해줍니다.

```python

  def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze()

```

여기서는 데이터가 모델을 통과하는 순서를 정의했습니다.

`out, _ = self.rnn(x)` : 입력 데이터 x가 RNN을 통과한 것입니다.

`self.fc(out[:, -1, :]).squeeze()`: 마지막 시점 출력값을 FC층에 통과시켰습니다.

---

#### Step 3. 학습시키기

```python

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

```

**코드 설명**

```python

criterion  = nn.MSELoss()
optimizer  = torch.optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 50

```

`criterion` : 평균 제곱 오차를 사용하는 오차 계산 함수입니다.

`optimizer` : Adam 최적화 알고리즘으로 가중치를 업데이트하는 방법을 뜻합니다. Adam 알고리즘은 오차를 줄이기 위해 가중치를 효율적으로 조정해주기 때문에 사용하게 되었습니다.

`lr=0.001` : 학습률을 뜻하며, 한 번에 가중치를 얼마나 조정할지 결정해줍니다.

`EPOCHS = 50` : 모델이 전체 데이터를 50번 학습한다는 것을 뜻합니다.

```python

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

```

50번 반복하면서 모델을 학습시키는 과정입니다.

`model.train()` : 학습 모드로 설정했습니다.

`pred = model(batch_X)` : 64개의 배치 데이터를 모델에 넣어서 발전량을 예측하는 과정입니다.

`loss = criterion(pred, batch_Y)` : 예측값과 실제값을 비교해서 오차를 계산하는 과정입니다.

`loss.backward()` : 역전파 과정입니다. 역전파란 오차를 줄이기 위해 가중치를 얼마나 조정해야 하는지 계산하는 과정입니다.

`optimizer.step()` :  `backward()`에서 계산된 방향으로 가중치를 업데이트하는 과정입니다.

이 모델이 50번동안 반복하는 과정은 이렇게 됩니다.

- 예측 -> 오차계산 -> 역전파 -> 가중치 업데이트 -> 다시 예측 ~..

```python

avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Test
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t.to(device))
        test_loss = criterion(test_pred, Y_test_t.to(device)).item()
        test_losses.append(test_loss)

```

`avg_train_loss = epoch_loss / len(train_loader)` : 평균 Train loss를 계산하는 과정입니다. (평균 Train loss = 배치별 loss를 합한 값 / 배치 수)

`model.eval()` : 평가 모드로 설정했습니다.

`with torch.no_grad()` : 평가 상황에서는 필요하지 않은 기울기를 계산하지 않는다는 뜻입니다.

`test_loss = criterion(test_pred, Y_test_t.to(device)).item()` : 예측값과 실제값의 오차를 계산하는 과정입니다.

---

#### Step 4. 성능 평가하기

```python

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

```

**코드 설명**

```python

model.eval()
with torch.no_grad():
    predicted_scaled = model(X_test_t.to(device)).cpu().numpy()

predicted = scaler_Y.inverse_transform(predicted_scaled.reshape(-1, 1))
actual    = scaler_Y.inverse_transform(Y_test.reshape(-1, 1))

```

`predicted_scaled = model(X_test_t.to(device)).cpu().numpy()` : 학습을 완료한 모델로 Test 데이터를 예측하는 과정입니다. 이때 값은 0~1로 정규화되어 나옵니다.

`predicted = scaler_Y.inverse_transform(predicted_scaled.reshape(-1, 1))` : 예측값(0~1)을 실제 발전량(kW)으로 역정규화하는 과정입니다.

`actual    = scaler_Y.inverse_transform(Y_test.reshape(-1, 1))` : 실제(0~1)을 실제 발전량(kW)으로 역정규화하는 과정입니다.

```python

mae  = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2   = r2_score(actual, predicted)

```

`mae  = mean_absolute_error(actual, predicted)` : 역정규화된 값으로 MAE를 계산하는 과정입니다.

**MAE(Mean Absolute Error)** 는 평균 절대 오차를 뜻하며, 예측값과 실제값 사이의 오차의 평균을 나타냅니다. MAE가 작을수록 모델 성능이 좋은 것입니다.

`rmse = np.sqrt(mean_squared_error(actual, predicted))` : 역정규화된 값으로 RMSE를 계산하는 과정입니다.

**RMSE(Root Mean Square Error)** 는 제곱근 평균 오차를 뜻하며, MAE에 비해 큰 오차에 더 민감하다는 특징이 있습니다. RMSE도 작을수록 모델 성능이 좋은 것입니다.

`r2   = r2_score(actual, predicted)` : 역정규화된 값으로 R²를 계산하는 과정입니다.

**R²(R-Squared)** 는 결정 계수를 뜻하며 1에 가까울 수록 모델 성능이 좋은 것입니다. 보통 0.7이상이면 양호한 편에 속한다고 판단합니다.

---

#### STEP 5. 결과 그래프 그리기

```python

# ============================================================
# Step 5: 결과 그래프
# ============================================================
print("\nStep 5: 그래프 생성 중...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Solar Power Prediction - RNN\nMAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}', fontsize=13)

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
plt.savefig(f'{output_dir}/rnn_result.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"그래프 저장 완료: {output_dir}/rnn_result.png")

```

**코드(그래프) 설명**

`fig, axes = plt.subplots(1, 2, figsize=(14, 5))` : 그래프 2개를 가로로 배치하는 코드입니다.

```python

# 실제 vs 예측
axes[0].plot(actual[:300],    label='Actual',    color='#2196F3', linewidth=1)
axes[0].plot(predicted[:300], label='Predicted', color='#F44336', linewidth=1, linestyle='--')
axes[0].set_title('Actual vs Predicted (First 300)')
axes[0].set_xlabel('Time Step')
axes[0].set_ylabel('Power (kW)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

```

왼쪽에 나올 그래프에 대한 내용입니다.

x축을 Test 데이터의 순서 번호로 설정하고 y축을 발전량(power_kW)으로 설정해 예측을 계속 할수록 예측값이랑 실제값이랑 차이가 어느 정도 나는지 확인하기 편하게 했습니다.

실제 발전량을 파란색으로 그리고 예측 발전량을 빨간색으로 그렸습니다.

또한 처음 300개의 데이터만 표시했습니다.

```python

# Train/Test Loss
axes[1].plot(train_losses, label='Train Loss', color='#2196F3')
axes[1].plot(test_losses,  label='Test Loss',  color='#F44336')
axes[1].set_title('Loss Curve')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss (MSE)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

```

오른쪽에 나올 그래프에 대한 내용입니다.

x축을 EPOCH로 설정하고 y축을 loss(기준: 평균 제곱 오차)로 설정해 EPOCH 별로  Train Loss와 Test Loss 변화를 측정했습니다.

---

### 1-3. RNN 모델 결과

<img width="940" height="380" alt="image" src="https://github.com/user-attachments/assets/bf2a7f58-3215-4c21-8ba8-d51bd704c969" />

<img width="940" height="450" alt="image" src="https://github.com/user-attachments/assets/ad9685b2-8838-4598-aaa6-b192d3ff5fa0" />



## 2. LSTM 모델 구축

### 2-1. LSTM 모델 설명

### 2-2. LSTM 모델 코딩

### 2-3. LSTM 모델 결과

## 3. Transformer 모델 구축

### 3-1. Transformer 모델 설명

### 3-2. Transformer 모델 코딩

### 3-3. Transformer 모델 결과

---

# V. Evaluation & Analysis

~~시작

---

# VI. Related Work (e.g., existing studies)

사용한 데이터셋 :  

NIST 사이트 (발전량 데이터) - https://pvdata.nist.gov/

NSRDB 사이트 (기상 상황 데이터) - https://nsrdb.nlr.gov/data-viewer

--

RNN 모델 설명 :



LSTM 모델 설명 :



Transformer 모델 설명 :



---

# VII. Conclusion: Discussion

~~시작

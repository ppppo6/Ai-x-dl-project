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

최근 탄소중립과 친환경 에너지에 대한 관심이 증가하면서 태양광 발전의 활용이 빠르게 확대되고 있습니다. 
 
그러나 태양광 발전은 일사량, 기온, 습도 풍속 등 다양한 기상 조건에 영향을 받기 때문에 발전량이 일정하지 않다는 한계가 있기 때문에 전력 수급의 안정화와 비용을 관리하기 위해서는 태양광 발전량을 예측하는 기술이 중요합니다.

본 프로젝트의 목표는 실제 태양광 발전량 데이터와 기상 데이터를 활용하여 태양광 발전량을 예측하는 딥러닝 모델을 구축하는 것입니다.
 

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
  (48 - 1일치 / 168 - 3.5일치)

**4. Train / Test 분할하기 (Train - 80% / Test - 20%)**

시계열 데이터의 특성상 시간 순서대로 데이터셋의 앞 80%를 Train, 뒤 20%를 Test로 분할해야 했습니다.

 [preprocessing_4.py](./preprocessing/preprocessing_4.py)

```python

"""코드 일부분""

# ============================================================
# Step 4: 시퀀스 생성
# ============================================================
print("\nStep 4: 시퀀스 생성 중...")
WINDOW_SIZE = 168  #3.5일치

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

```

---

# IV. Methodology

## 1. RNN 모델

### 1-1. RNN 모델 설명

### 1-2. RNN 모델 코딩

## 2. LSTM 모델

### 2-1. LSTM 모델 설명

### 2-2. LSTM 모델 코딩

## 3. Transformer 모델

### 3-1. Transformer 모델 설명

### 3-2. Transformer 모델 코딩

---

# V. Evaluation & Analysis

---

# VI. Related Work (e.g., existing studies)

사용한 데이터셋 :  

NIST 사이트 (발전량 데이터) - https://pvdata.nist.gov/

NSRDB 사이트 (기상 상황 데이터) - https://nsrdb.nlr.gov/data-viewer

RNN 모델 설명 :



LSTM 모델 설명 :



Transformer 모델 설명 :



---

# VII. Conclusion: Discussion

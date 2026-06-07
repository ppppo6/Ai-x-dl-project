
"""

NSRDB 기상 데이터 전처리

시간 컬럼 생성
결측치 제거
음수값 0으로 처리 (일사량)
소수점 반올림 (3자리)

"""

import pandas as pd

# ============================================================
# 경로 설정
# ============================================================
input_file  = 'C:/Users/doyun/OneDrive/Desktop/Proj/raw_data/1140304_39.13_-77.18_2017.csv'
output_file = 'C:/Users/doyun/OneDrive/Desktop/Proj/data_preprocessing/weatherdata/2017_weatherdata_final.csv'

# ============================================================
# Step 1: 데이터 로드 (헤더 2줄 건너뛰기)
# ============================================================
print("Step 1: 데이터 로드 중...")
df = pd.read_csv(input_file, skiprows=2)
print(f"원본 데이터: {len(df):,}행")
print(df.head())

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

# ============================================================
# Step 3: 결측치 제거
# ============================================================
print("\nStep 4: 결측치 제거 중...")
before = len(df)
df = df.dropna()
after = len(df)
print(f"결측치 제거: {before - after}행 제거")

# ============================================================
# Step 4: 음수 일사량 → 0 처리
# ============================================================
print("\nStep 5: 음수 일사량 처리 중...")
df['GHI'] = df['GHI'].clip(lower=0)
df['Clearsky GHI'] = df['Clearsky GHI'].clip(lower=0)
print("음수 일사량 → 0으로 처리 완료")

# ============================================================
# Step 5: 소수점 반올림 (3자리)
# ============================================================
print("\nStep 6: 소수점 반올림 중...")
numeric_cols = ['Temperature', 'Clearsky GHI', 'GHI',
                'Relative Humidity', 'Pressure', 'Wind Speed']
df[numeric_cols] = df[numeric_cols].round(3)
print("소수점 3자리로 반올림 완료")

# ============================================================
# Step 6: 저장
# ============================================================
df.to_csv(output_file, index=False)

print("\n" + "="*50)
print("✅ NSRDB 전처리 완료!")
print(f"총 행 수:  {len(df):,}개")
print(f"시작 시간: {df['timestamp'].iloc[0]}")
print(f"끝 시간:   {df['timestamp'].iloc[-1]}")
print(f"저장 위치: {output_file}")
print("="*50)

print("\n데이터 미리보기:")
print(df.head(10))
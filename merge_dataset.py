
"""

NIST 발전량 + NSRDB 기상 데이터 합치기

"""

import pandas as pd

# ============================================================
# 경로 설정
# ============================================================
power_file   = 'C:/Users/doyun/OneDrive/Desktop/Proj/data_preprocessing/PV.generation/2017_PV.generation_final.csv'
weather_file = 'C:/Users/doyun/OneDrive/Desktop/Proj/data_preprocessing/weatherdata/2017_weatherdata_final.csv'
output_file  = 'C:/Users/doyun/OneDrive/Desktop/Proj/data_preprocessing/merge_dataset/merged_dataset.csv'

# ============================================================
# Step 1: 데이터 로드
# ============================================================
print("Step 1: 데이터 로드 중...")
power   = pd.read_csv(power_file)
weather = pd.read_csv(weather_file)
print(f"발전량 데이터: {len(power):,}행")
print(f"기상 데이터:   {len(weather):,}행")

# ============================================================
# Step 2: timestamp 형식 통일
# ============================================================
print("\nStep 2: timestamp 형식 통일 중...")
power['timestamp']   = pd.to_datetime(power['timestamp'])
weather['timestamp'] = pd.to_datetime(weather['timestamp'])
print(f"발전량 시작: {power['timestamp'].iloc[0]}")
print(f"기상    시작: {weather['timestamp'].iloc[0]}")

# ============================================================
# Step 3: timestamp 기준으로 합치기
# ============================================================
print("\nStep 3: 데이터 합치는 중...")
df = pd.merge(power, weather, on='timestamp', how='inner')
print(f"합치기 완료: {len(df):,}행")

# ============================================================
# Step 4: 저장
# ============================================================
df.to_csv(output_file, index=False)

print("\n" + "="*50)
print("✅ 데이터 합치기 완료!")
print(f"총 행 수:  {len(df):,}개")
print(f"시작 시간: {df['timestamp'].iloc[0]}")
print(f"끝 시간:   {df['timestamp'].iloc[-1]}")
print(f"컬럼:      {list(df.columns)}")
print(f"저장 위치: {output_file}")
print("="*50)
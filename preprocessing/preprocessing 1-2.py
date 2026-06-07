
"""

NIST 발전량 데이터 전처리

타임존 제거 (UTC-5)
1분 단위 → 30분 단위 리샘플링
음수값 0으로 처리
결측치 제거
소수점 반올림 (3자리)

"""

import pandas as pd

# ============================================================
# 경로 설정
# ============================================================
input_file  = 'C:/Users/doyun/OneDrive/Desktop/Proj/data_preprocessing/PV.generation/merge_PV.generation/2017_PV.generation_merged.csv'
output_file = 'C:/Users/doyun/OneDrive/Desktop/Proj/data_preprocessing/PV.generation/2017_PV.generation_final.csv'

# ============================================================
# Step 1: 데이터 로드
# ============================================================
print("Step 1: 데이터 로드 중...")
df = pd.read_csv(input_file)
print(f"원본 데이터: {len(df):,}행")
print(df.head())

# ============================================================
# Step 2: 시간 형식 통일 (타임존 제거)
# ============================================================
print("\nStep 2: 시간 형식 통일 중...")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 타임존 제거 (-05:00 제거)
df['timestamp'] = df['timestamp'].dt.tz_localize(None)
print(f"시간 형식 변환 완료: {df['timestamp'].iloc[0]}")

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

# ============================================================
# Step 4: 음수값 → 0 처리 (야간 노이즈)
# ============================================================
print("\nStep 4: 음수값 처리 중...")
negative_count = (df_30min['power_kW'] < 0).sum()
df_30min['power_kW'] = df_30min['power_kW'].clip(lower=0)
print(f"음수값 {negative_count}개 → 0으로 처리")

# ============================================================
# Step 5: 결측치 제거
# ============================================================
print("\nStep 5: 결측치 제거 중...")
before = len(df_30min)
df_30min = df_30min.dropna()
after = len(df_30min)
print(f"결측치 제거: {before - after}행 제거")

# ============================================================
# Step 6: 소수점 반올림 (2자리)
# ============================================================
print("\nStep 6: 소수점 반올림 중...")
df_30min['power_kW'] = df_30min['power_kW'].round(3)
print("소수점 3자리로 반올림 완료")

# ============================================================
# Step 7: 저장
# ============================================================
df_30min.to_csv(output_file, index=False)

print("\n" + "="*50)
print("✅ NIST 전처리 완료!")
print(f"총 행 수:  {len(df_30min):,}개")
print(f"시작 시간: {df_30min['timestamp'].iloc[0]}")
print(f"끝 시간:   {df_30min['timestamp'].iloc[-1]}")
print(f"저장 위치: {output_file}")
print("="*50)

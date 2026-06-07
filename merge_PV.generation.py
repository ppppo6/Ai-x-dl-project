import pandas as pd
import glob
import os

# 경로 설정
data_folder = 'C:/Users/doyun/OneDrive/Desktop/Proj/raw_data/2017'  # 2017 폴더 경로
output_file = 'C:/Users/doyun/OneDrive/Desktop/Proj/data_preprocessing/PV.generation/merge_PV.generation/2017_PV.generation_merged.csv' #저장 경로

# 파일 찾기
all_files = sorted(glob.glob(os.path.join(data_folder, '**/*.csv'), recursive=True))
print(f"총 {len(all_files)}개 파일 발견")

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

print(f"\n✅ 완료! 총 {len(combined):,}행")
print(f"저장 위치: {output_file}")

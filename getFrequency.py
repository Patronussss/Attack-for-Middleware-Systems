import csv
from tqdm import tqdm
from pytrends.request import TrendReq
import os
import time
from datetime import datetime


diagnosis_list = []
output_file = "output.txt"
with open(output_file, 'r') as file:
    for line in file:
        # 去除每行末尾的换行符并添加到列表中
        diagnosis_list.append(line.strip())
pytrends = TrendReq(hl='en-US', tz=360)

# 设置时间范围为2009年1月至当前时间
start_date = '2009-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')
timeframe = start_date + ' ' + end_date

for kw in tqdm(diagnosis_list):
    file_path = f'frequency\{kw}.csv'
    
    if os.path.exists(file_path):
        print(f"File '{kw}.csv' already exists. Skipping...")
        continue
    
    try:
        kw_list = [kw]
        pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo='', gprop='')
        interest_over_time_df = pytrends.interest_over_time()
        interest_over_time_df.to_csv(file_path, index=True)
        # time.sleep(65)
    except Exception as e:
        print(f"Error processing keyword '{kw}': {str(e)}")

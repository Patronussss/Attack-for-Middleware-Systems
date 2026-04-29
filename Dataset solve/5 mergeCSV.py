import pandas as pd
import os
# 将当前文件夹下的csv文件合并
csv_directory = "E:/code/Searchable Encryption/dataset/PUDF/ZLDS/2017/Aux Data"

csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

combined_df = pd.DataFrame()

for file in csv_files:
    file_path = os.path.join(csv_directory, file)
    print(file_path)
    df = pd.read_csv(file_path, low_memory=False)
    combined_df = pd.concat([combined_df, df], ignore_index=True)

combined_df.to_csv("E:/code/Searchable Encryption/dataset/PUDF/ZLDS/2017/2017_withoutQ4.csv", index = False)


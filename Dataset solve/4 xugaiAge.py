import os
import pandas as pd

# 指定要处理的文件夹路径
folder_path = "E:/code/Searchable Encryption/dataset/PUDF/ZLDS/2019"

# 遍历文件夹及子文件夹中的所有文件
for dirpath, _, filenames in os.walk(folder_path):
    for filename in filenames:
        if filename.endswith('.csv'):
            # 构建文件的完整路径
            file_path = os.path.join(dirpath, filename)
            
            # 读取 CSV 文件
            df = pd.read_csv(file_path)
            
            # 修改 Age 列，给每个值后加上 ' year'
            if 'Age' in df.columns:
                df['Age'] = df['Age'].astype(str) + ' year'
                
                # 保存修改后的 DataFrame 回 CSV 文件
                df.to_csv(file_path, index=False)
                print(f"已修改文件: {file_path}")

print("所有文件处理完成！")
# import os
# import pandas as pd

# # 指定要处理的文件夹路径
# folder_path = 'F:\\Desktop\\Supplementary experiments\\New Dataset\\4q2010'

# # 遍历文件夹及子文件夹中的所有文件
# for dirpath, _, filenames in os.walk(folder_path):
#     for filename in filenames:
#         if filename.endswith('.csv'):
#             # 构建文件的完整路径
#             file_path = os.path.join(dirpath, filename)
            
            # 读取 CSV 文件
#             df = pd.read_csv(file_path)
            
#             # 修改 Age 列，去掉多余的 'year'
#             if 'Age' in df.columns:
#                 df['Age'] = df['Age'].str.replace(r'\s*year\s*year$', ' year', regex=True)
                
#                 # 保存修改后的 DataFrame 回 CSV 文件
#                 df.to_csv(file_path, index=False)
#                 print(f"已修改文件: {file_path}")

# print("所有文件处理完成！")

# file_path = 'F:\\Desktop\\Supplementary experiments\\New Dataset\\4q2010\\text_5000.csv'
# df = pd.read_csv(file_path)

# # 修改 Age 列，去掉多余的 'year'
# # 处理Age列
# df['Age'] = df['Age'].astype(str) + ' year'

# df.to_csv(file_path, index=False)
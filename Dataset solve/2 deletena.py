import pandas as pd
# 读取CSV文件


csv_file = "E:/code/Searchable Encryption/dataset/PUDF/ZLDS/2019/PUDF_base1_1q2019.csv"
df = pd.read_csv(csv_file)

# df = pd.read_csv('F:\\code\\Searchable Encryption\\dataset\\PUDF\\Result\\PUDF_base2q2010-without lengthOfStay.csv')

# 检测空白行并删除
df.dropna(how='any', inplace=True)

# 将清理后的数据保存回CSV文件
df.to_csv(csv_file,index=False)
# df.to_csv('F:\\code\\Searchable Encryption\\dataset\\PUDF\\Result\\PUDF_base2q2010-without lengthOfStay.csv', index=False)
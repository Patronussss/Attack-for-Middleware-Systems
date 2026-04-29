import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('F:/Desktop/Supplementary experiments/result/TIFS/dataset/2010_plain.csv')

base = ['Age', 'Admission Type', 'Length of stay', 'Risk','Gender', 'Race','Hospital','Pincipal Diagnosis']
# 统计每列数据的分布
for column in df.columns:
    if column not in base:
        continue
    print(f"列名: {column}")
    print(f"数据类型: {df[column].dtype}")
    
    # 数值型数据
    if pd.api.types.is_numeric_dtype(df[column]):
        print("统计描述:")
        print(df[column].describe())
        
        # 绘制直方图
        plt.figure(figsize=(10, 6))
        df[column].hist(bins=20)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()
    
    # 分类型数据
    else:
        print("值计数:")
        print(df[column].value_counts())
        
        # 绘制条形图
        plt.figure(figsize=(10, 6))
        df[column].value_counts().plot(kind='bar')
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.savefig(f'F:/Desktop/Supplementary experiments/result/TIFS/Figure/2010_plain_{column}_distribution.pdf', format='pdf')
        plt.show()
    
    print("\n" + "="*50 + "\n")
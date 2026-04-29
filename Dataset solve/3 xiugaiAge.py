import csv


def process_age_column(csv_file):
    """
    从CSV文件中提取Age列,处理后重新输出.

    参数:
    csv_file (str) - CSV文件的路径
    """
    # 读取CSV文件
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)

    # 提取Age列并处理
    for row in data:
        age = row['Age']
        if isinstance(age, str):
            if 'days' in age:
                row['Age'] = 0
            elif '+' in age:
                row['Age'] = int(age.split('+')[0])
            elif '-' in age:
                row['Age'] = int(age.split('-')[0])
            else:
                row['Age'] = int(age)

    # 将处理后的数据写回CSV文件
    with open(csv_file, 'w', newline='') as output_file:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    print(f"已将处理后的CSV文件输出到 'processed_{csv_file}'.")


csv_file1 = "E:/code/Searchable Encryption/dataset/PUDF/ZLDS/2019/PUDF_base1_1q2019.csv"
process_age_column(csv_file1)
csv_file2 = "E:/code/Searchable Encryption/dataset/PUDF/ZLDS/2019/PUDF_base1_2q2019.csv"
process_age_column(csv_file2)
csv_file3 = "E:/code/Searchable Encryption/dataset/PUDF/ZLDS/2019/PUDF_base1_3q2019.csv"
process_age_column(csv_file3)
csv_file4 = "E:/code/Searchable Encryption/dataset/PUDF/ZLDS/2019/PUDF_base1_4q2019.csv"
process_age_column(csv_file4)
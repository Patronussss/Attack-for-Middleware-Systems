import csv

def extract_and_process(csv_file_path, column_index):
    data = set()
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) > column_index:
                    cell_data = row[column_index].strip().lower()
                    if cell_data:
                        data.add(cell_data)
    except FileNotFoundError:
        print(f"错误：未找到文件 {csv_file_path}。")
    except Exception as e:
        print(f"发生未知错误：{e}")
    data_count = len(data)
    return data, data_count

def write_to_txt(data, txt_file_path):
    try:
        with open(txt_file_path, 'w', encoding='utf-8') as file:
            for value in data:
                file.write(value + '\n')
        print(f"数据已成功写入 {txt_file_path}")
    except Exception as e:
        print(f"写入文件时出现错误: {e}")

# 示例使用
if __name__ == "__main__":
    csv_file_path = "E:/code/Searchable Encryption/dataset/Alzheimer_s_Disease_and_Healthy_Aging_Data.csv"
    column_index = 21  # 假设提取第 1 列数据，索引从 0 开始
    data, data_count = extract_and_process(csv_file_path, column_index)
    print(f"数据组数：{data_count}")
    print(data)
    # write_to_txt(data, "F:/Desktop/Supplementary experiments/Alzheimer.txt")
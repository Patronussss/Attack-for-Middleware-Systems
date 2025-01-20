import csv
from tqdm import tqdm
import random
import pandas as pd
import os 
import re
from collections import Counter

# record_count = [] # 记录数
# record_recovered_count = [] #  成功恢复的记录数
# keyword_count = [] # 关键字个数
# keyword_recovered_count = [] # 成功恢复的关键字数

# 根据表头选择特定列生成子矩阵
def generate_submatrix(matrix, header, columns):
    header_indices = [matrix[0].index(col) for col in columns]
    submatrix = []  
    for row in matrix[1:]:
        subrow = [row[i] for i in header_indices]
        submatrix.append(subrow)
    return submatrix

# 创建字典记录行号和对应的record id列的值
def create_rowid_dict(matrix, header, record_id_column):
    rowid_dict = {}
    record_id_index = header.index(record_id_column)
    for index, row in enumerate(matrix[1:], start=1):
        rowid_dict[index] = row[record_id_index]
    return rowid_dict

# 读取CSV文件并生成矩阵
def read_csv_to_matrix(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            matrix.append(row)
    return matrix

# # 定义一个函数，用于统计矩阵中出现的关键字数量
# def count_keywords(matrix):
#     # 创建一个空集合，用于存储所有出现的关键字
#     keywords = set()
#     # 遍历矩阵中的所有元素
#     for row in matrix:
#         for element in row:
#             # 使用正则表达式提取关键字
#             keywords.update(re.findall(r"\b\w+\b", element))
#     # 返回关键字的数量
#     return len(keywords)

def count_keywords(matrix):
    keyword_set = set()
    for row in matrix:
        for element in row:
            keyword_set.add(element)
    return len(keyword_set)

# 定义一个函数，用来计算二元矩阵转十进制序列
def binary_to_decimal(matrix):
    # 调换矩阵的行列
    transposed_matrix = matrix.T
    # 将每一行的值组合起来转为十进制
    decimal_values = []
    for row in transposed_matrix:
        decimal_value = int("".join(map(str, row)), 2)
        decimal_values.append(decimal_value)
    return decimal_values

def find_unique_rows_withNone(matrix):
   # 使用字典来存储行数据及其出现次数
    row_count = {}
    unique_rows = []

    for row in matrix:
        serial_number = row[0]
        row_data = tuple(row[1:])

        # 检查当前行是否包含 None
        if None not in row_data:
            if row_data in row_count:
                row_count[row_data].append(serial_number)
            else:
                row_count[row_data] = [serial_number]

    # 找出仅出现一次的行
    for row_data, serial_numbers in row_count.items():
        if len(serial_numbers) == 1:
            unique_rows.extend(serial_numbers)

    return unique_rows

def find_unique_rows(matrix):
    # 使用字典来存储行的出现次数
    row_count = {}

    # 遍历每一行，从第二列开始将数据组合成一个元组
    for row in matrix:
        row_data = tuple(row[1:])  # 从第二列开始组合
        if row_data in row_count:
            row_count[row_data] += 1
        else:
            row_count[row_data] = 1

    # 找出仅出现一次的行
    unique_rows = [list(row) for row, count in row_count.items() if count == 1]
    return unique_rows

def replace_values_with_none(matrix, reference_dict):
    # 遍历每一行，从第二列开始检查
    for row in matrix:
        for i in range(1, len(row)):
            if row[i] not in reference_dict:
                row[i] = None
    return matrix

def generate_decimal_sequence(matrix):

    # 将矩阵转为DataFrame
    df = pd.DataFrame(matrix[1:], columns=matrix[0])  # 忽略第一行作为列名

    # 删除第一列（序列号），只保留关键字
    df_keywords = df.drop(columns=[df.columns[0]])

    # 为每个关键字生成二进制矩阵
    binary_matrix = pd.get_dummies(df_keywords.apply(lambda x: pd.Series(x)).stack()).groupby(level=0).sum()

    # 确保列名匹配关键字
    keywords = sorted(set(df_keywords.values.flatten()))
    binary_matrix = binary_matrix.reindex(columns=keywords, fill_value=0)

    # 将每一列转化为二进制字符串，并转化为十进制数
    decimal_sequence = [int(''.join(map(str, binary_matrix[keyword])), 2) for keyword in keywords]
    
    return decimal_sequence, keywords

index_list = [1,2,3,4]

for il in index_list:

    base = 'F:/Desktop/text/'+ str(il)+ 'q2010/'
    target = 'F:/Desktop/text/'+ str(il)+ 'q2010/'
    out = 'F:/Desktop/text/'+ str(il)+ 'q2010 output of A1_new'
    lists = [i[:-4] for i in os.listdir(out)]
    file_list = os.listdir(base)
    for file_name in sorted(file_list):
        # if file_name[:-4] not in lists:
            print(file_name)
            print("---------------------------")
            filePath = os.path.join(base,file_name)
            filePathPlain = os.path.join(target,file_name)

            matrix_cipher = read_csv_to_matrix(filePath)
            record_id_mapping = create_rowid_dict(matrix_cipher, matrix_cipher[0], 'Record ID') # 记录行置换情况

            selected_columns_ope_det_sse_withoutid = [ 'Age', 'Admission Type', 'Length of stay', 'Risk','Discharge', 'Gender', 'Race','Hospital', 'Pincipal Diagnosis']

            matrix_ods_withoutid = generate_submatrix(matrix_cipher, matrix_cipher[0], selected_columns_ope_det_sse_withoutid)

            selected_columns_ope_det_sse = ['Record ID', 'Age', 'Admission Type', 'Length of stay', 'Risk','Discharge', 'Gender', 'Race','Hospital', 'Pincipal Diagnosis']

            matrix_cipher_ods = generate_submatrix(matrix_cipher, matrix_cipher[0], selected_columns_ope_det_sse)

            keyword_count = count_keywords(matrix_ods_withoutid) # 一共有多少个关键字
            # unique_rows = find_unique_rows(matrix_cipher_ods)
            # record_recovered_count = len(unique_rows)
            record_count = len(matrix_cipher) - 1 # 一共有多少条数据

            # 计算OPE+DET的关键字数量
            selected_columns_ope = ['Record ID', 'Age', 'Admission Type', 'Length of stay', 'Risk']
            matrix_cipher_ope = generate_submatrix(matrix_cipher, matrix_cipher[0], selected_columns_ope)
            def custom_sort(row):
                return tuple(int(cell) if cell.isdigit() else cell for cell in row[1:])

            # 按照要求对矩阵进行排序
            sorted_matrix_cipher = sorted(matrix_cipher_ope, key=custom_sort)

            # 读取原始CSV文件
            with open(filePath, 'r') as infile:
                reader = csv.reader(infile)
                header = next(reader)  # 保留表头
                data = [row for row in reader]

            # 随机打乱行顺序
            random.shuffle(data)

            with open(filePathPlain, 'w', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(header)  # 写入表头
                writer.writerows(data)

            # print("CSV文件行顺序已随机打乱并写入新文件中。")


            matrix_plain = read_csv_to_matrix(filePathPlain)
            record_id_mapping_plain = create_rowid_dict(matrix_plain, matrix_plain[0], 'Record ID') # 记录行置换情况

            matrix_plain_ope = generate_submatrix(matrix_plain, matrix_plain[0], selected_columns_ope)

            # 按照要求对矩阵进行排序
            sorted_matrix_plain = sorted(matrix_plain_ope, key=custom_sort)

            value_mapping = {}

            for i in range(len(sorted_matrix_cipher)):
                row_cipher = sorted_matrix_cipher[i][1:]  # 去掉第一列
                row_plain = sorted_matrix_plain[i][1:]   # 去掉第一列
                if row_cipher == row_plain:
                    # matching_pairs.append((sorted_matrix_cipher[i][0], sorted_matrix_plain[i][0]))  # 记录第一列的值
                    for j in range(len(row_cipher)):
                        value_mapping[row_cipher[j]] = row_plain[j]

            print(f"在经过OPE后恢复了{len(value_mapping)}个数据")
            print(f"OPE一共有{count_keywords(matrix_cipher_ope)-record_count}个数据")
            print("-----------OPE-------------")

            selected_columns_det = ['Record ID', 'Discharge', 'Gender', 'Race']
            matrix_cipher_det = generate_submatrix(matrix_cipher, matrix_cipher[0], selected_columns_det)

            element_counts = {}  # 用于存储元素及其出现次数的字典
            for col_index in range(1, len(matrix_cipher_det[0])):  # 从第二列开始（索引为1）
                column_data = [row[col_index] for row in matrix_cipher_det]  # 提取该列的数据
                for element in column_data:
                    if element in element_counts:
                        element_counts[element] += 1
                    else:
                        element_counts[element] = 1
            # print(element_counts)

            matrix_plain_det = generate_submatrix(matrix_plain, matrix_plain[0], selected_columns_det)

            element_counts_plain = {}  # 用于存储元素及其出现次数的字典
            for col_index in range(1, len(matrix_plain_det[0])):  # 从第二列开始（索引为1）
                column_data = [row[col_index] for row in matrix_plain_det]  # 提取该列的数据
                for element in column_data:
                    if element in element_counts_plain:
                        element_counts_plain[element] += 1
                    else:
                        element_counts_plain[element] = 1
            # print(element_counts_plain)

            for key1, value1 in element_counts.items():
                for key2, value2 in element_counts_plain.items():
                    if value1 == value2:
                        value_mapping[key1] = key2

            selected_columns_ope_det = ['Record ID','Age', 'Admission Type', 'Length of stay', 'Risk','Discharge', 'Gender', 'Race']

            matrix_cipher_ope_det = generate_submatrix(matrix_cipher, matrix_cipher[0], selected_columns_ope_det)

            # unique_rows = find_unique_rows(matrix_cipher_ope_det)
            # record_recovered_count = len(unique_rows) # OPE + DET 可以恢复的行数

            print(f"在经过OPE和DET后恢复了{len(value_mapping)}个数据")
            print(f"OPE和DET一共有{count_keywords(matrix_cipher_ope_det)-record_count}个数据")
            # 从500开始，DET中就总有两个关键字无法恢复（应该是频率恰好一样）
            print("------------DET--------------")
            # 看看能恢复多少行
            matrix_od_replaced = replace_values_with_none(matrix_cipher_ope_det, value_mapping)

            unique_rows_replaced = find_unique_rows_withNone(matrix_od_replaced)
            # unique_rows_noreplace = find_unique_rows(matrix_cipher_ope_det)

            # print(len(unique_rows_noreplace))
            # print(len(unique_rows_replaced))
            
            # 尽管不能恢复全部的OPE+DET，但是OPE+DET独特行可以全部恢复
            # 开始SSE

            selected_columns_sse = ['Record ID', 'Hospital','Pincipal Diagnosis']
            data_dict_cipher = {}
            stop_words = ["and", "of", "or", "for", "with", "to", "not", "by", "in", "the", "but", "from", "as"]

            matrix_cipher_sse = generate_submatrix(matrix_cipher, matrix_cipher[0], selected_columns_sse)

            for row in matrix_cipher_sse:
                record_id = row[0]
                for i in range(1, len(row)):
                    word = row[i]
                    if word.lower() not in stop_words:
                        if word in data_dict_cipher:
                            data_dict_cipher[word].append(record_id)
                        else:
                            data_dict_cipher[word] = [record_id]
            # 先要拿到前面恢复的记录的行数
            # recovered_id = list(unique_rows_replaced.values())

            matrix_recovered = []

            total_iterations = len(matrix_cipher_sse)
            with tqdm(total=total_iterations, desc="generating matrix recovered") as pbar:
                for row in matrix_cipher_sse:
                    if row[0] in unique_rows_replaced:
                        matrix_recovered.append(row)
                    pbar.update(1)
            
            # 生成两个十进制序列及其对应的关键字
            decimal_sequence1, keywords1 = generate_decimal_sequence(matrix_cipher_sse)
            decimal_sequence2, keywords2 = generate_decimal_sequence(matrix_cipher_sse)

            # 创建一个映射，将十进制值映射回关键字
            decimal_to_keyword1 = {decimal: keyword for decimal, keyword in zip(decimal_sequence1, keywords1)}
            decimal_to_keyword2 = {decimal: keyword for decimal, keyword in zip(decimal_sequence2, keywords2)}

            # 找出每个序列中独特的值（只出现一次的值）
            unique_values1 = [decimal for decimal in decimal_sequence1 if decimal_sequence1.count(decimal) == 1]
            unique_values2 = [decimal for decimal in decimal_sequence2 if decimal_sequence2.count(decimal) == 1]

            # 找出两个序列中相同的独特值
            common_unique_values = set(unique_values1) & set(unique_values2)

            # 记录这些相同的值对应的关键字
            common_keywords = {decimal_to_keyword1[value]: decimal_to_keyword2[value] for value in common_unique_values if value in decimal_to_keyword1 and value in decimal_to_keyword2}
            value_mapping.update(common_keywords)


            print(f"在经过volume后恢复了{len(value_mapping)}个数据")
            keyword_count = count_keywords(matrix_cipher_ods)-record_count
            print(keyword_count)
            matrix_s_replaced = replace_values_with_none(matrix_cipher_ods, value_mapping)

            unique_rows_s_replaced = find_unique_rows_withNone(matrix_s_replaced)

            print("-------------SSE-------------")
            print(len(unique_rows_s_replaced))
            unique = find_unique_rows(matrix_cipher_ods)
            print(len(unique))
            print(record_count)

            # 已经把能恢复的行全恢复了，现在看怎么恢复更多的关键字
            keyword_volume = {}
            for key, value in data_dict_cipher.items():
                keyword_volume[key] = len(value)
            
            # 使用 Counter 统计所有 value 的出现次数
            value_counts = Counter(keyword_volume.values())

            # 使用列表推导获取所有独一无二的值
            unique_values = [v for v, count in value_counts.items() if count == 1]

            # 使用 setdefault 方法更新 value_mapping
            _ = [value_mapping.setdefault(value, [value]) for value in unique_values]

            values = {}
            for key, value in value_mapping.items():
                if key == value:
                    values[key] = value
            
            # output = "/Users/cherry/Desktop/zengli/output of A1/"
            file_extension = os.path.splitext(file_name)[1]
            new_file_name = file_name.replace(file_extension, ".txt")
            outputPath = os.path.join(out,new_file_name)
            print(outputPath)
            with open(outputPath,"w") as f:
                f.write("record number: " + str(record_count) + " successfully recovered number: " + str(len(unique)) + " keywords number: " + str(keyword_count) + " successfully recovered keyword number: " + str(len(values)))


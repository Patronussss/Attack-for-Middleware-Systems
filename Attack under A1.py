import sys
sys.path.append("F:/Desktop/Supplementary experiments") 
import functions
import csv
from tqdm import tqdm
import pandas as pd
import os 
import re
import random
import numpy as np
from collections import Counter
import math

def euclidean_distance(dict1, dict2):
    distance = 0
    if len(dict1) != len(dict2):
        all_keys = set(dict1.keys()).union(dict2.keys())  # 获取两个字典的所有键
        sorted_keys = sorted(all_keys)  # 对键进行排序

        
        values1 = [dict1.get(key, 0) for key in sorted_keys]  # 提取第一个字典的值，如果键不存在则填充为0
        values2 = [dict2.get(key, 0) for key in sorted_keys]  # 提取第二个字典的值，如果键不存在则填充为0

        # 计算这两个列表之间的欧氏距离
        distance = math.sqrt(sum((value1 - value2) ** 2 for value1, value2 in zip(values1, values2)))
    else:
        values1 = sorted(dict1.values())
        values2 = sorted(dict2.values())

        # 计算这两个列表之间的欧氏距离
        distance = math.sqrt(sum((value1 - value2) ** 2 for value1, value2 in zip(values1, values2)))

    return distance

def select_rows_and_generate_matrix(num_rows, csv_file):
    # 用于存储最终选择的行，包括表头
    selected_rows = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        # 将CSV文件的所有行读取到一个列表中
        all_rows = list(reader)
        header = all_rows[0]
        data_rows = all_rows[1:]
        
        # 检查请求的行数是否超过CSV文件中的总行数（不包括表头）
        if num_rows > len(data_rows):
            raise ValueError("请求的行数超过CSV文件中的总行数")
        selected_data_rows = random.sample(data_rows, num_rows)
        selected_rows.append(header)
        selected_rows.extend(selected_data_rows)
    return selected_rows

def write_dicts_to_txt(column_count, recovered_count, output_file):
    with open(output_file, 'w') as file:
        for key in column_count.keys():
            line = f"{key}: {column_count[key]} columns need recovery, {recovered_count[key]} columns successfully recovered\n"
            file.write(line)

def generate_submatrix(matrix, columns):
    # 获取表头
    header = matrix[0]
    # 找到选定列的索引
    header_indices = [header.index(col) for col in columns]
    # 创建子矩阵并保留表头
    submatrix = [columns]  # 添加新的表头
    for row in matrix[1:]:
        subrow = [row[i] for i in header_indices]
        submatrix.append(subrow)
    return submatrix

# 读取CSV文件并生成矩阵
def read_csv_to_matrix(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            matrix.append(row)
    return matrix

def shuffle_matrix(matrix):
    df = pd.DataFrame(matrix)
    df.columns = matrix[0]
    new_matrix = df[1:].take(np.random.permutation(len(matrix[0])),axis=1).take(np.random.permutation(len(df[1:])),axis=0)
    new_matrix1 = new_matrix.values.tolist()
    new_matrix = [new_matrix.columns.values.tolist()] + new_matrix1
    return new_matrix

def count_column_elements(matrix):
    header = matrix[0]
    result = {}
    for col_idx in range(0, len(matrix[0])):
        col_name = header[col_idx]
        col_freq = {}
        if col_name == "Hospital" or col_name == "Pincipal Diagnosis":
            col_freq = {}
        else:
            for row in matrix[1:]:
                element = row[col_idx]
                if element in col_freq:
                    col_freq[element] += 1
                else:
                    col_freq[element] = 1
        result[col_name] = col_freq
    return result

def is_unique_value(dictionary, value):
    value_count = 0
    for val in dictionary.values():
        if val == value:
            value_count += 1
            if value_count > 1:
                return False
    return True

# 提取指定行数的子矩阵作为敌手拿到的明文信息
def random_submatrix(matrix, num_rows):
    header = matrix[0]
    data = matrix[1:]
    
    random_rows = random.sample(data, num_rows)
    submatrix = [header] + random_rows
    
    return submatrix

def convert_nested_counts_to_frequencies(nested_count_dict):
    frequency_dict = {}
    
    for col_name, sub_dict in nested_count_dict.items():
        total_count = sum(sub_dict.values())
        frequency_dict[col_name] = {key: value / total_count for key, value in sub_dict.items()}
    
    return frequency_dict

filePathPlain = "F:/Desktop/Attack for Datablinder/2015/PUDF_base1_1q2015.csv"
matrixP = functions.read_csv_to_matrix(filePathPlain)

index_list = [1,2,3,4]
for il in index_list:
    base = 'F:\\Desktop\\Supplementary experiments\\New Dataset\\'+ str(il)+ 'q2010/'
    file_list = os.listdir(base)
    for file_name in sorted(file_list):
        for time in range(1,6):
            out = 'F:/Desktop/Supplementary experiments/result/A2/'+ str(il)+ 'q2010 output of A2/' + str(time) + '/'
            if not os.path.exists(out):
                os.makedirs(out)
            filePath = os.path.join(base,file_name)
            matrix = functions.read_csv_to_matrix(filePath)

            selected_columns_cipher = ['Age', 'Admission Type', 'Length of stay', 'Risk','Discharge', 'Gender', 'Race','Hospital', 'Pincipal Diagnosis']
            selected_columns_plain = ['Age', 'Admission Type', 'Length of stay', 'Risk','Discharge', 'Gender', 'Race']

            matrix_temp = generate_submatrix(matrix, selected_columns_cipher)
            matrix_plain = generate_submatrix(matrixP, selected_columns_plain)

            matrix_cipher =  shuffle_matrix(matrix_temp)

            unique_elements_cipher = count_column_elements(matrix_cipher)
            unique_elements_plain = count_column_elements(matrix_plain)

            unique_counts_cipher = {}
            for key,value in unique_elements_cipher.items():
                count = len(value)
                unique_counts_cipher[key] = count
            unique_counts_plain = {}
            for key,value in unique_elements_plain.items():
                count = len(value)
                unique_counts_plain[key] = count

            result = {}
            for key1, value1 in unique_counts_plain.items():
                if is_unique_value(unique_counts_plain, value1):
                    for key2, value2 in unique_counts_cipher.items():
                        if value2 == value1 and is_unique_value(unique_counts_cipher, value2):
                            result[key2] = key1

            unique_frequency_cipher = convert_nested_counts_to_frequencies(unique_elements_cipher)
            unique_frequency_plain = convert_nested_counts_to_frequencies(unique_elements_plain)

            for key1, value1 in unique_counts_plain.items():
                # print(key1, value1)
                # print("=========")
                if key1 not in result.keys():
                    dict1 = unique_frequency_plain[key1]
                    # print(dict1)
                    for key2, value2 in unique_counts_cipher.items():
                        # print(key2, value2)
                        # print("---------")
                        if key2 not in result.values() and  abs(value1 - value2) < 500:
                            dict2 = unique_frequency_cipher[key2]
                            # print(dict2)
                            if dict2:
                                dis = euclidean_distance(dict1, dict2)
                                # print(key1, key2, dis)
                                if dis < 0.1:
                                    result[key2] = key1
                                    break
            OPE_list = ['Age', 'Admission Type', 'Length of stay', 'Risk']
            DET_list = ['Discharge', 'Gender', 'Race']
            countOPE = 0
            countDET = 0
            # print(result)
            
            for key, value in result.items():
                if key == value and key in OPE_list:
                    countOPE += 1
                elif key == value and key in DET_list:
                    countDET += 1
            column_count = len(unique_elements_plain)
            recovered_count = countDET + countOPE

            file_extension = os.path.splitext(file_name)[1]
            new_file_name = file_name.replace(file_extension, ".txt")
            outputPath = os.path.join(out,new_file_name)
            print(outputPath)
            with open(outputPath,"w") as f:
                f.write("keywords number: " + str(column_count) + " successfully recovered keyword number: " + str(recovered_count))


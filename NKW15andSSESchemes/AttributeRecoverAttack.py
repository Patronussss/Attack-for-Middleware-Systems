import sys
sys.path.append("/media/ices/machenrry/zl/Attack for DataBlinder/") 
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
        all_keys = set(dict1.keys()).union(dict2.keys())
        sorted_keys = sorted(all_keys)

        values1 = [dict1.get(key, 0) for key in sorted_keys]
        values2 = [dict2.get(key, 0) for key in sorted_keys]

        distance = math.sqrt(sum((value1 - value2) ** 2 for value1, value2 in zip(values1, values2)))
    else:
        values1 = sorted(dict1.values())
        values2 = sorted(dict2.values())

        distance = math.sqrt(sum((value1 - value2) ** 2 for value1, value2 in zip(values1, values2)))

    return distance

def select_rows_and_generate_matrix(num_rows, csv_file):
    selected_rows = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        all_rows = list(reader)
        header = all_rows[0]
        data_rows = all_rows[1:]
        
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
    header = matrix[0]
    header_indices = [header.index(col) for col in columns]
    submatrix = [columns]
    for row in matrix[1:]:
        subrow = [row[i] for i in header_indices]
        submatrix.append(subrow)
    return submatrix

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

def AttributeRecoverAttack(matrix_cipher, matrix_plain, selected_columns_cipher, selected_columns_plain):
    matrix_cipher_shuffled = shuffle_matrix(matrix_cipher)
    matrix_plain_sub = generate_submatrix(matrix_plain, selected_columns_plain)

    unique_elements_cipher = count_column_elements(matrix_cipher_shuffled)
    unique_elements_plain = count_column_elements(matrix_plain_sub)

    unique_counts_cipher = {}
    for key, value in unique_elements_cipher.items():
        count = len(value)
        unique_counts_cipher[key] = count
    unique_counts_plain = {}
    for key, value in unique_elements_plain.items():
        count = len(value)
        unique_counts_plain[key] = count

    result = {}
    for key1, value1 in unique_counts_cipher.items():
        if is_unique_value(unique_counts_cipher, value1):
            for key2, value2 in unique_counts_plain.items():
                if value2 == value1 and is_unique_value(unique_counts_plain, value2):
                    result[key1] = key2
                    break
        else:
            matching_keys = []
            for key2, value2 in unique_counts_plain.items():
                if value2 == value1:
                    matching_keys.append(key2)
            if matching_keys:
                result[key1] = matching_keys

    OPE_list = ['Age', 'Admission Type', 'Length of stay', 'Risk']
    DET_list = ['Discharge', 'Gender', 'Race']
    countOPE = 0
    countDET = 0

    for key, value in result.items():
        if isinstance(value, list):
            if key in value and key in OPE_list:
                countOPE += 1
            elif key in value and key in DET_list:
                countDET += 1
        else:
            if key == value and key in OPE_list:
                countOPE += 1
            elif key == value and key in DET_list:
                countDET += 1

    column_count = len(unique_elements_plain)
    recovered_count = countDET + countOPE

    return result, recovered_count, column_count

if __name__ == '__main__':
    out = 'result/A2-NKW15/oringinal/' 
    root = "dataset/text_508029.csv"

    base = [500, 725, 1050, 1525, 2210, 3205, 4645, 6735, 9765, 14160, 20530, 29770, 43170, 62600, 90750, 131600, 190850, 276750, 401300, 508029]
    matrix = functions.read_csv_to_matrix(root)

    if not os.path.exists(out):
        os.makedirs(out)

    for ind in base:
        print(ind)
        recovered_keywords = []
        keywords_count = []
        for _ in range(50):
            matrix_cipher = functions.random_extract(matrix, ind)
            
            selected_columns_cipher = ['Age', 'Admission Type', 'Length of stay', 'Risk', 'Discharge', 'Gender', 'Race', 'Hospital', 'Pincipal Diagnosis']
            selected_columns_plain = ['Age', 'Admission Type', 'Length of stay', 'Risk', 'Discharge', 'Gender', 'Race']
            num_rows_to_extract = int((len(matrix_cipher)-1) * 0.9)
            matrixP = functions.random_submatrix(matrix_cipher, num_rows_to_extract)
            matrix_temp = generate_submatrix(matrix, selected_columns_cipher)

            result, recovered_count, column_count = AttributeRecoverAttack(
                matrix_temp, matrixP, selected_columns_cipher, selected_columns_plain
            )

            print(result)
            recovered_keywords.append(recovered_count)
            keywords_count.append(column_count)

        avg_keyword_number = sum(keywords_count) / 50
        avg_recovered_keywords = sum(recovered_keywords) / 50

        new_file_name = "text_" + str(ind) + ".txt"
        
        outputPath = os.path.join(out, new_file_name)
        print(outputPath)
        with open(outputPath, "w") as f:
            f.write("keywords number: " + str(avg_keyword_number) + " successfully recovered keyword number: " + str(avg_recovered_keywords))

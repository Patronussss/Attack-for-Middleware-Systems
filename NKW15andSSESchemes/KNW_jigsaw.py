import sys
sys.path.append("/media/ices/machenrry/zl/Attack for DataBlinder/") 
import functions
import csv
from tqdm import tqdm
import pandas as pd
import re
import random
import numpy as np
from collections import Counter
from scipy.optimize import linear_sum_assignment
from datetime import datetime
import os
from FrequencyAnalysisAttack import FrequencyAnalysisAttack
from AttributeRecoverAttack import AttributeRecoverAttack

def find_closest_mapping(dict1, dict2):
    mapping = {}
    for key1, value1 in dict1.items():
        min_distance = np.inf
        candidate = key1
        for key2, value2 in dict2.items():
            distance = abs(value1 - value2)
            if distance < min_distance:
                min_distance = distance
                candidate = key2
        mapping[key1] = candidate
    return mapping

def replace_values_with_none(matrix, reference_dict):
    for row in matrix:
        for i in range(1, len(row)):
            if row[i] not in reference_dict:
                row[i] = "zlzlzl"
    return matrix

def read_csv_to_matrix(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            matrix.append(row)
    return matrix

def generate_submatrix(matrix, columns):
    header = matrix[0]
    header_indices = [header.index(col) for col in columns]
    submatrix = [columns]
    for row in matrix[1:]:
        subrow = [row[i] for i in header_indices]
        submatrix.append(subrow)
    return submatrix

def count_keywords(matrix):
    keywords = set()
    for row in matrix:
        for element in row:
            keywords.update(re.findall(r"\b[\w\s]+\b", element))
    return len(keywords)

def custom_sort(value):
    try:
        return int(value)
    except ValueError:
        return value

def extract_columns(matrix, target_column):
    headers = matrix[0]
    index = headers.index(target_column)
    return [row[index] for row in matrix[1:]]

def all_numeric(input_list):
    return all(item.isdigit() for item in input_list)

def compute_frequency_and_cdf(data):
    freq = Counter(data)
    total = sum(freq.values())
    cdf = {}
    cumulative = 0

    for key in sorted(freq):
        count = freq[key]
        cumulative += count
        freq[key] = (count / total) * 10000
        cdf[key] = (cumulative / total) * 10000

    return freq, cdf

def column_frequencies(matrix):
    matrix = np.array(matrix)
    frequencies = {}
    
    for col_index in range(matrix.shape[1]):
        col = matrix[:, col_index]
        unique, counts = np.unique(col, return_counts=True)
        total = len(col)
        freq_dict = {str(key): round(count / total, 5) for key, count in zip(unique, counts)}
        frequencies[col_index] = freq_dict
    
    return frequencies

def find_optimal_mapping(data_c, data_z):
    freq_c, cdf_c = compute_frequency_and_cdf(data_c)
    freq_z, cdf_z = compute_frequency_and_cdf(data_z)

    keys_c = sorted(freq_c.keys(), key=custom_sort)
    keys_z = sorted(freq_z.keys(), key=custom_sort)

    cost_matrix = np.zeros((len(keys_c), len(keys_z)))

    for i, key_c in enumerate(keys_c):
        for j, key_z in enumerate(keys_z):
            freq_diff = (freq_c[key_c] - freq_z[key_z]) ** 2
            cdf_diff = (cdf_c[key_c] - cdf_z[key_z]) ** 2
            cost_matrix[i, j] = freq_diff + cdf_diff
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {keys_c[i]: keys_z[j] for i, j in zip(row_ind, col_ind)}
    return mapping

def find_unique_value(input_list):
    value_counts = {}
    
    for value in input_list:
        if value in value_counts:
            value_counts[value] += 1
        else:
            value_counts[value] = 1
    
    unique_values = [key for key, count in value_counts.items() if count == 1]
    return unique_values

def create_rowid_dict(matrix, header, record_id_column):
    rowid_dict = {}
    record_id_index = header.index(record_id_column)
    
    for index, row in enumerate(matrix[1:], start=1):
        record_id = row[record_id_index]
        rowid_dict[record_id] = index
    
    return rowid_dict

def count_frequency_multicol(matrix, column_names):
    num_cols = matrix.shape[1]
    frequency_dicts = [{} for _ in range(num_cols)]

    for col_idx in range(num_cols):
        column = matrix[:, col_idx]
        total_count = len(column)

        for key in column:
            key_str = str(key)
            frequency_dicts[col_idx][key_str] = frequency_dicts[col_idx].get(key_str, 0) + 1 / total_count

    final_dict = {}

    for col_idx, freq_dict in enumerate(frequency_dicts):
        col_name = column_names[col_idx]
        final_dict[col_name] = freq_dict

    return final_dict

def convert_to_date(year_month):
    year, month = map(int, year_month.split('.'))
    return datetime(year, month, 1)

data = ['2019.9-2019.12', '2020.1-2020.4', '2020.5-2020.8', '2020.9-2020.12',
        '2021.1-2021.4', '2021.5-2021.8', '2021.9-2021.12',
        '2022.1-2022.4', '2022.5-2022.8', '2022.9-2022.12',
        '2023.1-2023.4', '2023.5-2023.8', '2023.9-2023.12',
        '2024.1-2024.4', '2024.5-2024.8', '2024.9-2024.12']

def find_period(year_month):
    target_date = convert_to_date(year_month)
    
    for idx, period in enumerate(data):
        start_year_month, end_year_month = period.split('-')
        start_date = convert_to_date(start_year_month)
        end_date = convert_to_date(end_year_month)
        
        if start_date <= target_date <= end_date:
            return period

def process_csv_file(file_path):
    result_dict = {}
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            time_period = row[0]
            frequency = int(row[1])
            times = time_period.split("-")
            year = times[0]
            month = times[1]
            t = f'{year}.{month}'
            key = find_period(t)
            if key in result_dict:
                result_dict[key] += frequency
            else:
                result_dict[key] = frequency
    return result_dict

def numToFreqency(data_dict):
    total_counts = {key: sum(sub_dict.values()) for key, sub_dict in data_dict.items()}
    total_frequency = sum(total_counts.values())
    data_dict1 = {}
    data_dict2 = {}
    for key, sub_dict in data_dict.items():
        sub_dict1 = {}
        freq = 0
        if not bool(sub_dict):
            continue
        else:
            for period, count in sub_dict.items():
                sub_dict1[period] = count / total_frequency
                freq += count
            data_dict1[key] = sub_dict1
            data_dict2[key] = freq / total_frequency
    return data_dict1, data_dict2

index_list = [4]

filePathPlain = "dataset/2015.csv"
matrixP = functions.read_csv_to_matrix(filePathPlain)

root = "dataset/text_508029.csv"
base = [500, 725, 1050, 1525, 2210, 3205, 4645, 6735, 9765, 14160, 20530, 29770, 43170, 62600, 90750, 131600, 190850, 401300, 508029]

matrix = functions.read_csv_to_matrix(root)
out = 'result/A4/A4_KNW24/'
if not os.path.exists(out):
        os.makedirs(out)

for i in base:
    print(i)
    
    recovered_keywords = []
    keywords_count = []
    for _ in range(50):
        matrix_c = functions.random_extract(matrix, i)

        selected_columns = ['Record ID', 'Age', 'Admission Type', 'Length of stay', 'Risk', 'Gender', 'Race','Hospital', 'Pincipal Diagnosis']
        selected_columns_noid = ['Age', 'Admission Type', 'Length of stay', 'Risk', 'Gender', 'Race','Hospital', 'Pincipal Diagnosis']

        matrix_cipher = functions.generate_submatrix(matrix_c, selected_columns)
        matrix_plain = functions.generate_submatrix(matrixP, selected_columns)

        selected_columns_plain_for_attr_recovery = ['Age', 'Admission Type', 'Length of stay', 'Risk', 'Discharge', 'Gender', 'Race']
        
        cipher_columns_all = matrix_cipher[0][1:]
        matrix_cipher_for_attr_recovery = functions.generate_submatrix(matrix_cipher, ['Record ID'] + cipher_columns_all)
        matrix_plain_for_attr_recovery = functions.generate_submatrix(matrix_plain, selected_columns_plain_for_attr_recovery)
        
        attr_mapping, _, _ = AttributeRecoverAttack(
                matrix_cipher_for_attr_recovery, 
                matrix_plain_for_attr_recovery, 
                ['Record ID'] + cipher_columns_all, 
                selected_columns_plain_for_attr_recovery
            )

        selected_column_ope = ['Record ID']
        ope_target_cols = ['Age', 'Admission Type', 'Length of stay', 'Risk']
        for cipher_col, plain_col in attr_mapping.items():
            if plain_col in ope_target_cols:
                selected_column_ope.append(cipher_col)

        selected_columns_det = []
        det_target_cols = ['Gender', 'Race']
        for cipher_col, plain_col in attr_mapping.items():
            if plain_col in det_target_cols:
                selected_columns_det.append(cipher_col)

        selected_column_sse = []
        sse_target_cols = ['Hospital', 'Pincipal Diagnosis']
        for cipher_col, plain_col in attr_mapping.items():
            if plain_col in sse_target_cols:
                selected_column_sse.append(cipher_col)

        selected_columns_od = ['Record ID']
        od_target_cols = ['Age', 'Admission Type', 'Length of stay', 'Risk', 'Gender', 'Race']
        for cipher_col, plain_col in attr_mapping.items():
            if plain_col in od_target_cols:
                selected_columns_od.append(cipher_col)

        matrix_cipher_ope = functions.generate_submatrix(matrix_cipher, selected_column_ope)
        matrix_plain_ope = functions.generate_submatrix(matrix_plain, selected_column_ope)
        value_mapping = {}

        ope_cols_mapping = {}
        for i, col in enumerate(selected_column_ope[1:], start=1):
            for cipher_col, plain_col in attr_mapping.items():
                if cipher_col == col:
                    ope_cols_mapping[col] = plain_col
                    break

        for ope_col in selected_column_ope[1:]:
            plain_col_name = ope_cols_mapping.get(ope_col, ope_col)
            column_cipher = functions.extract_columns(matrix_cipher_ope, ope_col)
            cipher_count = Counter(column_cipher)
            keyword_count_cipher = len(cipher_count)

            column_plain = functions.extract_columns(matrix_plain_ope, ope_col)
            plain_count = Counter(column_plain)
            keyword_count_plain = len(plain_count)

            if keyword_count_cipher == keyword_count_plain:
                sorted_cipher = sorted(set(column_cipher))
                sorted_plain = sorted(set(column_plain))
                mapping = {key: value for key, value in zip(sorted_cipher, sorted_plain)}
                value_mapping.update(mapping)
            else:
                if plain_col_name == 'Length of stay':
                    sorted_cipher = sorted(column_cipher, key=functions.custom_sort)
                    sorted_plain = sorted(column_plain, key=functions.custom_sort)
                    mapping = functions.find_optimal_mapping(sorted_cipher, sorted_plain)
                else:
                    mapping = functions.find_optimal_mapping(column_cipher, column_plain)
                value_mapping.update(mapping)

        keyword_count_stay = 0
        for c, p in attr_mapping.items():
            if p == 'Length of stay':
                keyword_count_stay = functions.count_keywords(functions.generate_submatrix(matrix_cipher, [c])[1:])
                break

        keyword_count_aar = 0
        for col in ['Age', 'Admission Type', 'Risk']:
            cipher_col = None
            for c, p in attr_mapping.items():
                if p == col:
                    cipher_col = c
                    break
            if cipher_col:
                keyword_count_aar += functions.count_keywords(functions.generate_submatrix(matrix_cipher, [cipher_col])[1:])

        print(f"length of stay: {keyword_count_stay}")
        print(f"Age + Admission Type + Risk: {keyword_count_aar}")

        mapping_det, time_det, accuracy_det, keyword_count_det, count_det = FrequencyAnalysisAttack(matrix_cipher, matrix_plain, selected_columns_det)
        value_mapping.update(mapping_det)
        
        value_mapping_od = {}
        for key, value in value_mapping.items():
            if key == value:
                value_mapping_od[key] = value

        matrix_cipher_od = functions.generate_submatrix(matrix_cipher, selected_columns_od)
        matrix_recovered = functions.replace_values_with_none(matrix_cipher_od[1:], value_mapping_od)

        matrix_plain_od = functions.generate_submatrix(matrix_plain, selected_columns_od)
        list_cipher_od = []
        record_cipher = []
        for row in matrix_recovered[1:]:
            if None not in row[1:]:
                res = ' '.join(row[1:])
                record_cipher.append(row[0])
                list_cipher_od.append(res)
        frequency_cipher_od  = Counter(list_cipher_od)

        list_plain_od = []
        record_plain = []
        for row in matrix_plain_od[1:]:
            res = ' '.join(row[1:])
            record_plain.append(row[0])
            list_plain_od.append(res)
        frequency_plain_od = Counter(list_plain_od)

        unique_values_cipher_od = functions.find_unique_value(list_cipher_od)
        unique_values_plain_od = functions.find_unique_value(list_plain_od)

        unique_rows = [v for v in unique_values_cipher_od if v in unique_values_plain_od]

        recovered_rows = {}
        for row in tqdm(unique_rows, desc="Processing rows", total=len(unique_rows)):
            id_cipher = record_cipher[list_cipher_od.index(row)]
            id_plain = record_plain[list_plain_od.index(row)]
            recovered_rows[id_cipher] = id_plain

        print(len(recovered_rows))

        recordid_cipher = functions.create_rowid_dict(matrix_cipher, matrix_cipher[0], 'Record ID')
        recordid_plain = functions.create_rowid_dict(matrix_plain, matrix_plain[0], 'Record ID')

        for id_c, id_p in recovered_rows.items():
            row_cipher = matrix_cipher[recordid_cipher[id_c]]
            index_plain = recordid_plain[id_p]
            row_plain = matrix_plain[index_plain]
            for i in range(1,len(row_cipher)):
                value_mapping[row_cipher[i]] = row_plain[i]

        for key, value in value_mapping.items():
            if key == value:
                value_mapping_od[key] = value

        matrix_cipher_sse = functions.generate_submatrix(matrix_cipher, selected_column_sse)
        matrix_plain_sse = functions.generate_submatrix(matrix_plain, selected_column_sse)

        volumn_cipher_sse = functions.count_frequency_multicol(np.array(matrix_cipher_sse[1:]), matrix_cipher_sse[0])
        volumn_plain_sse = functions.count_frequency_multicol(np.array(matrix_plain_sse[1:]), matrix_plain_sse[0])
        mapping_sse = {}
        for key, dict1 in volumn_cipher_sse.items():
            dict2 = volumn_plain_sse[key]
            temp = functions.find_closest_mapping(dict1, dict2)
            value_mapping.update(temp)
            mapping_sse.update(temp)

        for key, value in mapping_sse.items():
            if key == value:
                value_mapping_od[key] = value
            
        keyword_number = functions.count_keywords(functions.generate_submatrix(matrix_cipher, selected_columns_noid)[1:])

        hospital_cipher_col = None
        diagnosis_cipher_col = None
        for c, p in attr_mapping.items():
            if p == 'Hospital':
                hospital_cipher_col = c
            elif p == 'Pincipal Diagnosis':
                diagnosis_cipher_col = c

        hospital_plain = functions.extract_columns(matrix_plain, hospital_cipher_col) if hospital_cipher_col else functions.extract_columns(matrix_plain, 'Hospital')
        hospital_cipher = functions.extract_columns(matrix_cipher, hospital_cipher_col) if hospital_cipher_col else functions.extract_columns(matrix_cipher, 'Hospital')
        hospital_list = list(set(hospital_plain).union(set(hospital_cipher)))
        
        diagnosis_plain = functions.extract_columns(matrix_plain, diagnosis_cipher_col) if diagnosis_cipher_col else functions.extract_columns(matrix_plain, 'Pincipal Diagnosis')
        diagnosis_cipher = functions.extract_columns(matrix_cipher, diagnosis_cipher_col) if diagnosis_cipher_col else functions.extract_columns(matrix_cipher, 'Pincipal Diagnosis')
        diagnosis_list = list(set(diagnosis_plain).union(set(diagnosis_cipher)))

        list_dh = hospital_list + diagnosis_list

        hospital_frequency_dict = {}

        frequency_folder = '/media/ices/machenrry/zl/Attack for DataBlinder/frequency/'
        for value in hospital_list:
            csv_file_path = os.path.join(frequency_folder, f'{value}.csv')
            if os.path.exists(csv_file_path):
                value_dict = functions.process_csv_file(csv_file_path)
                hospital_frequency_dict[value] = value_dict

        hos_freq_cipher_dict = {key: value for key,value in hospital_frequency_dict.items() if key in hospital_cipher}
        hos_freq_plain_dict = {key: value for key,value in hospital_frequency_dict.items() if key in hospital_plain}
        diagnosis_frequency_dict = {}

        for value in diagnosis_list:
            csv_file_path = os.path.join(frequency_folder, f'{value}.csv')
            if os.path.exists(csv_file_path):
                value_dict = functions.process_csv_file(csv_file_path)
                diagnosis_frequency_dict[value] = value_dict

        diag_freq_cipher_dict = {key: value for key,value in diagnosis_frequency_dict.items() if key in diagnosis_cipher}
        diag_freq_plain_dict = {key: value for key,value in diagnosis_frequency_dict.items() if key in diagnosis_plain}

        hos_volume_cipher_dict = volumn_cipher_sse.get(hospital_cipher_col, volumn_cipher_sse.get('Hospital', {}))
        diag_volume_cipher_dict = volumn_cipher_sse.get(diagnosis_cipher_col, volumn_cipher_sse.get('Pincipal Diagnosis', {}))
        hos_volume_plain_dict = volumn_plain_sse.get(hospital_cipher_col, volumn_plain_sse.get('Hospital', {}))
        diag_volume_plain_dict = volumn_plain_sse.get(diagnosis_cipher_col, volumn_plain_sse.get('Pincipal Diagnosis', {}))

        hos_freq_cipher_dict1, hos_freq_cipher_dict_noT = functions.numToFreqency(hos_freq_cipher_dict)
        hos_freq_plain_dict1, hos_freq_plain_dict_noT = functions.numToFreqency(hos_freq_plain_dict)
        diag_freq_cipher_dict1, diag_freq_cipher_dict_noT = functions.numToFreqency(diag_freq_cipher_dict)
        diag_freq_plain_dict1, diag_freq_plain_dict_noT = functions.numToFreqency(diag_freq_plain_dict)

        dis_dict_ = {}
        alpha = 0.3
        for key1 in set(hospital_cipher):
            if key1 in hos_volume_cipher_dict and key1 in hos_freq_cipher_dict_noT:
                volume1 = hos_volume_cipher_dict[key1]
                freq1 = hos_freq_cipher_dict_noT[key1]
                min_score = 100
                for key2 in set(hospital_cipher):
                    if key2 != key1 and key2 in hos_volume_cipher_dict and key2 in hos_freq_cipher_dict_noT:
                        volume2 = hos_volume_cipher_dict[key2]
                        freq2 = hos_freq_cipher_dict_noT[key2]
                        v = abs(volume1 - volume2) * alpha
                        f = abs(freq1 - freq2) * (1-alpha)
                        score = v + f
                        if score < min_score:
                            min_score = score
                dis_dict_[key1] = min_score
        for key1 in set(diagnosis_cipher):
            if key1 in diag_volume_cipher_dict and key1 in diag_freq_cipher_dict_noT:
                volume1 = diag_volume_cipher_dict[key1]
                freq1 = diag_freq_cipher_dict_noT[key1]
                min_score = 100
                for key2 in set(diagnosis_cipher):
                    if key2 != key1 and key2 in diag_volume_cipher_dict and key2 in diag_freq_cipher_dict_noT:
                        volume2 = diag_volume_cipher_dict[key2]
                        freq2 = diag_freq_cipher_dict_noT[key2]
                        v = abs(volume1 - volume2) * alpha
                        f = abs(freq1 - freq2) * (1-alpha)
                        score = v + f
                        if score < min_score:
                            min_score = score
                dis_dict_[key1] = min_score
        sorted_dis_dict_ = dict(sorted(dis_dict_.items(), key=lambda item: item[1], reverse=True))
        print(len(set(hospital_cipher)))
        print(len(sorted_dis_dict_))

        pred_dict_ = {}
        for key1, value in sorted_dis_dict_.items():
            if key1 in hos_volume_cipher_dict and key1 in hos_freq_cipher_dict_noT:
                volume1 = hos_volume_cipher_dict[key1]
                freq1 = hos_freq_cipher_dict_noT[key1]
                min_score = 100
                candidate = key1
                for key2 in set(hospital_plain):
                    if key2 in hos_volume_plain_dict and key2 in hos_freq_plain_dict_noT:
                        volume2 = hos_volume_plain_dict[key2]
                        freq2 = hos_freq_plain_dict_noT[key2]
                        v = abs(volume1 - volume2) * alpha
                        f = abs(freq1 - freq2) * (1-alpha)
                        score = v + f
                        if score < min_score:
                            min_score = score
                            candidate = key2
                pred_dict_[key1] = candidate
            else:
                volume1 = diag_volume_cipher_dict[key1]
                freq1 = diag_freq_cipher_dict_noT[key1]
                min_score = 100
                candidate = key1
                for key2 in set(diagnosis_plain):
                    if key2 in diag_volume_plain_dict and key2 in diag_freq_plain_dict_noT:
                        volume2 = diag_volume_plain_dict[key2]
                        freq2 = diag_freq_plain_dict_noT[key2]
                        v = abs(volume1 - volume2) * alpha
                        f = abs(freq1 - freq2) * (1-alpha)
                        score = v + f
                        if score < min_score:
                            min_score = score
                            candidate = key2
                pred_dict_[key1] = candidate

        count = 0
        for key,value in pred_dict_.items():
            if key == value:
                value_mapping_od[key] = value
                count += 1
        print(count)
        print(len(value_mapping_od))
        recovered_keywords.append(len(value_mapping_od))
        keywords_count.append(keyword_number)
    avg_keyword_number = sum(keywords_count) / 50
    avg_recovered_keywords = sum(recovered_keywords) / 50

    new_file_name = "text_" + str(i) + ".txt"
    outputPath = os.path.join(out,new_file_name)
    print(outputPath)
    with open(outputPath,"w") as f:
        f.write("keywords number: " + str(avg_keyword_number) + " successfully recovered keyword number: " + str(avg_recovered_keywords))

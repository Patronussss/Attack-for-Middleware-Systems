import sys
sys.path.append("/CCS2026/") 
import functions
import csv
from tqdm import tqdm
import random
from collections import Counter
import numpy as np
import os
from collections import defaultdict
import math
from scipy.optimize import linear_sum_assignment as hungarian
from NKW15andSSEschemes.FrequencyAnalysisAttack import FrequencyAnalysisAttack
from NKW15andSSESchemes.AttributeRecoverAttack import AttributeRecoverAttack

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

index_list = [5]
for il in index_list:
    
    if il == 4:
        out = 'result/other dataset/2010Q4/'
        filePathPlain = "dataset/2015.csv"
        root = "dataset/text_508029.csv"
    elif il == 5:
        out = 'result/CCS/A4_jigsaw_Alzheimer/'
        filePathPlain = "dataset/Alzheimer_plain.csv"
        root = "dataset/Alzheimer_cipher.csv"

    base = [500, 11816, 23132, 34448, 45764, 57080, 68396, 79712, 91028, 102344, 113657]
    matrixP = functions.read_csv_to_matrix(filePathPlain)
    
    matrix = functions.read_csv_to_matrix(root)

    selected_columns_plain_for_attr_recovery = ['YearStart', 'LocationAbbr', 'Stratification2', 'Class', 'DataValueTypeID', 'Topic', 'Low_Confidence_Limit', 'High_Confidence_Limit']

    if not os.path.exists(out):
        os.makedirs(out)
    for ind in base:
        print(ind)
        print("------------------------------")
        recovered_keywords = []
        keywords_count = []
        for _ in range(50):
            matrix_c = functions.random_extract(matrix, ind)

            cipher_columns_all = matrix_c[0][1:]
            
            matrix_cipher_for_attr_recovery = functions.generate_submatrix(matrix_c, ['Record ID'] + cipher_columns_all)
            matrix_plain_for_attr_recovery = functions.generate_submatrix(matrixP, selected_columns_plain_for_attr_recovery)
            
            attr_mapping, _, _ = AttributeRecoverAttack(matrix_cipher_for_attr_recovery, matrix_plain_for_attr_recovery, ['Record ID'] + cipher_columns_all, selected_columns_plain_for_attr_recovery)
            
            ope_target_cols = ['YearStart', 'LocationAbbr', 'Stratification2', 'Low_Confidence_Limit', 'High_Confidence_Limit']
            det_target_cols = ['Class', 'DataValueTypeID']
            sse_target_cols = ['Topic']
            
            selected_columns_ope = [cipher_col for cipher_col, plain_col in attr_mapping.items() if plain_col in ope_target_cols]
            selected_columns_det = [cipher_col for cipher_col, plain_col in attr_mapping.items() if plain_col in det_target_cols]
            selected_columns_sse = [cipher_col for cipher_col, plain_col in attr_mapping.items() if plain_col in sse_target_cols]
            
            selected_columns = selected_columns_ope + selected_columns_det + selected_columns_sse

            matrix_cipher = functions.generate_submatrix(matrix_c, selected_columns)
            matrix_plain = functions.generate_submatrix(matrixP, selected_columns)

            matrix_cipher_ope = functions.generate_submatrix(matrix_cipher, selected_columns_ope)
            matrix_plain_ope = functions.generate_submatrix(matrix_plain, selected_columns_ope)
            value_mapping = {}

            for ope_col in selected_columns_ope:
                plain_col = attr_mapping.get(ope_col, ope_col)
                column_cipher = functions.extract_columns(matrix_cipher_ope, ope_col)
                column_plain = functions.extract_columns(matrix_plain_ope, plain_col)
                
                keyword_count_cipher = len(Counter(column_cipher))
                keyword_count_plain = len(Counter(column_plain))

                if keyword_count_cipher == keyword_count_plain:
                    sorted_cipher = sorted(set(column_cipher))
                    sorted_plain = sorted(set(column_plain))
                    mapping = {key: value for key, value in zip(sorted_cipher, sorted_plain)}
                    value_mapping.update(mapping)
                else:
                    mapping = functions.find_optimal_mapping(column_cipher, column_plain)
                    value_mapping.update(mapping)

            mapping_det, time_det, accuracy_det, keyword_count_det, count_det = FrequencyAnalysisAttack(matrix_cipher, matrix_plain, selected_columns_det)
            value_mapping.update(mapping_det)
            
            selected_column_sse = selected_columns_sse
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
                    value_mapping[key] = value
                
            keyword_number = functions.count_keywords(functions.generate_submatrix(matrix_cipher, selected_columns)[1:])

            hospital_plain = functions.extract_columns(matrix_plain, selected_column_sse[0])
            hospital_cipher = functions.extract_columns(matrix_cipher, selected_column_sse[0])
            hospital_list = list(set(hospital_plain).union(set(hospital_cipher)))

            hospital_frequency_dict = {}

            frequency_folder = '/CCS2026/frequency/'
            for value in hospital_list:
                csv_file_path = os.path.join(frequency_folder, f'{value}.csv')
                if os.path.exists(csv_file_path):
                    value_dict = functions.process_csv_file(csv_file_path)
                    hospital_frequency_dict[value] = value_dict

            hos_freq_cipher_dict = {key: value for key,value in hospital_frequency_dict.items() if key in hospital_cipher}
            hos_freq_plain_dict = {key: value for key,value in hospital_frequency_dict.items() if key in hospital_plain}

            hos_volume_cipher_dict = volumn_cipher_sse[selected_column_sse[0]]
            hos_volume_plain_dict = volumn_plain_sse[selected_column_sse[0]]

            hos_freq_cipher_dict1, hos_freq_cipher_dict_noT = functions.numToFreqency(hos_freq_cipher_dict)
            hos_freq_plain_dict1, hos_freq_plain_dict_noT = functions.numToFreqency(hos_freq_plain_dict)

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
                            v = abs(volume1 - volume2) *alpha
                            f = abs(freq1 - freq2)*(1-alpha)
                            score = v+f
                            if score < min_score:
                                min_score = score
                    dis_dict_[key1] = min_score

            sorted_dis_dict_ = dict(sorted(dis_dict_.items(), key=lambda item: item[1], reverse=True))

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
                            v = abs(volume1 - volume2) *alpha
                            f = abs(freq1 - freq2)*(1-alpha)
                            score = v+f
                            if score < min_score:
                                min_score = score
                                candidate = key2
                    pred_dict_[key1] = candidate

            count = 0
            for key,value in pred_dict_.items():
                if key == value:
                    value_mapping[key] = value
                    count += 1
            print(count)
            print(len(value_mapping))
            recovered_keywords.append(len(value_mapping))
            keywords_count.append(keyword_number)
        avg_keyword_number = sum(keywords_count) / 50
        avg_recovered_keywords = sum(recovered_keywords) / 50

        new_file_name = "text_" + str(ind) + ".txt"
        outputPath = os.path.join(out,new_file_name)
        print(outputPath)
        with open(outputPath,"w") as f:
            f.write("keywords number: " + str(avg_keyword_number) + " successfully recovered keyword number: " + str(avg_recovered_keywords))

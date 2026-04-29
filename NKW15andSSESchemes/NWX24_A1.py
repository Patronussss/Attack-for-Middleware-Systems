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
from FrequencyAnalysisAttack import FrequencyAnalysisAttack
from datetime import datetime
import os
from AttributeRecoverAttack import AttributeRecoverAttack

def get_sub_matrix(V_cipher, cipher_list, knownQ):
    column_indices = [cipher_list.index(keyword) for keyword in knownQ if keyword in cipher_list]
    sub_matrix = []
    for row in V_cipher:
        sub_row = [row[index].item() for index in column_indices]
        sub_matrix.append(sub_row)
    return sub_matrix

root = "dataset/text_508029.csv"
base = [187500]
matrix = functions.read_csv_to_matrix(root)
frac_list = [0.9]

for ind in base:
    for frac in frac_list:
        for time in range(1,51):
            out = 'result/A3/NWX24/' + str(frac) + '/' + str(time) + '/'
            if not os.path.exists(out):
                os.makedirs(out)
            matrix_c = functions.random_extract(matrix, ind)

            selected_columns = ['Record ID', 'Age', 'Admission Type', 'Length of stay', 'Risk', 'Gender', 'Race','Hospital', 'Pincipal Diagnosis']
            selected_columns_noid = ['Age', 'Admission Type', 'Length of stay', 'Risk', 'Gender', 'Race','Hospital', 'Pincipal Diagnosis']
            matrix_cipher = functions.generate_submatrix(matrix_c, selected_columns)
            num_rows_to_extract =  int((len(matrix_cipher)-1) * frac)
            matrix_plain = functions.random_submatrix(matrix_cipher, num_rows_to_extract)

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

            keyword_count_stay = functions.count_keywords(functions.generate_submatrix(matrix_cipher, [
                attr_mapping.get('Length of stay', 'Length of stay')
            ])[1:]) 

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

            kw_c = keyword_number
            kw_s_c = len(value_mapping_od)

            new_file_name = "text_" + str(ind) + ".txt"
            outputPath = os.path.join(out, new_file_name)
            print(outputPath)
            with open(outputPath,"w") as f:
                f.write(" keywords number: " + str(kw_c) + " successfully recovered keyword number: " + str(kw_s_c) + "percent of file" + str(frac))

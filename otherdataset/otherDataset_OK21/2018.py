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

def process_matrix_from_mapping(matrix, columns, replacement_dict):
    header = matrix[0]
    column_indices = [header.index(col) for col in columns]

    for row in matrix[1:]:
        for index in column_indices:
            value = row[index]
            if value in replacement_dict and replacement_dict[value] == value:
                continue
            else:
                row[index] = 'zlzlzl'
    return matrix

filePathPlain = "dataset/2015.csv"
matrix_plain = functions.read_csv_to_matrix(filePathPlain)
out = 'result/CCS/A4_Oya21_2018/'
root = "dataset/PUDF_base1_4q2018.csv"
base = [500, 11816, 23132, 34448, 45764, 57080, 68396, 79712, 91028, 102344, 113657]

matrix = functions.read_csv_to_matrix(root)
matrixP = functions.read_csv_to_matrix(filePathPlain)

selected_columns_plain_for_attr_recovery = ['Age', 'Admission Type', 'Length of stay', 'Risk', 'Discharge', 'Gender', 'Race', 'Hospital', 'Pincipal Diagnosis']

if not os.path.exists(out):
    os.makedirs(out)

for i in base:
    print(i)
    recovered_keywords = []
    keywords_count = []
    for _ in range(5):
        matrix_c = functions.random_extract(matrix, i)

        cipher_columns_all = matrix_c[0][1:]
        
        matrix_cipher_for_attr_recovery = functions.generate_submatrix(matrix_c, ['Record ID'] + cipher_columns_all)
        matrix_plain_for_attr_recovery = functions.generate_submatrix(matrixP, selected_columns_plain_for_attr_recovery)
        
        attr_mapping, _, _ = AttributeRecoverAttack(matrix_cipher_for_attr_recovery, matrix_plain_for_attr_recovery, ['Record ID'] + cipher_columns_all, selected_columns_plain_for_attr_recovery)
        
        ope_target_cols = ['Age', 'Admission Type', 'Length of stay', 'Risk']
        det_target_cols = ['Gender', 'Race']
        sse_target_cols = ['Hospital', 'Pincipal Diagnosis']
        
        selected_columns_ope = ['Record ID'] + [cipher_col for cipher_col, plain_col in attr_mapping.items() if plain_col in ope_target_cols]
        selected_columns_det = [cipher_col for cipher_col, plain_col in attr_mapping.items() if plain_col in det_target_cols]
        selected_columns_sse = [cipher_col for cipher_col, plain_col in attr_mapping.items() if plain_col in sse_target_cols]
        selected_columns = ['Record ID'] + selected_columns_ope[1:] + selected_columns_det + selected_columns_sse

        matrix_cipher = functions.generate_submatrix(matrix_c, selected_columns)
        matrix_plain = functions.generate_submatrix(matrixP, selected_columns)

        matrix_cipher_ope = functions.generate_submatrix(matrix_cipher, selected_columns_ope)
        matrix_plain_ope = functions.generate_submatrix(matrix_plain, selected_columns_ope)
        value_mapping = {}

        for ope_col in selected_columns_ope[1:]:
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
        
        value_mapping_od = {}
        for key, value in value_mapping.items():
            if key == value:
                value_mapping_od[key] = value

        selected_columns_od = ['Record ID'] + selected_columns_ope[1:] + selected_columns_det

        matrix_cipher_od = functions.generate_submatrix(matrix_cipher, selected_columns_od)
        matrix_recovered = functions.replace_values_with_none(matrix_cipher_od[1:], value_mapping_od)

        matrix_plain_od = functions.generate_submatrix(matrix_plain, selected_columns_od)
        list_cipher_od = []
        record_cipher = []
        for row in matrix_recovered:
            if None not in row[1:]:
                res = ' '.join(row[1:])
                record_cipher.append(row[0])
                list_cipher_od.append(res)
        frequency_cipher_od = Counter(list_cipher_od)

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
            for idx in range(1, len(row_cipher)):
                value_mapping[row_cipher[idx]] = row_plain[idx]

        for key, value in value_mapping.items():
            if key == value:
                value_mapping_od[key] = value

        selected_column_sse = selected_columns_sse
        matrix_cipher_sse = functions.generate_submatrix(matrix_cipher, selected_column_sse)
        matrix_plain_sse = functions.generate_submatrix(matrix_plain, selected_column_sse)

        hospital_cipher = functions.extract_columns(matrix_cipher, selected_column_sse[0])
        hospital_cipher_list = list(set(hospital_cipher))

        hospital_frequency_cipher_dict = {}

        frequency_folder = '/CCS2026/frequency/'
        for value in hospital_cipher_list:
            csv_file_path = os.path.join(frequency_folder, f'{value}.csv')
            if os.path.exists(csv_file_path):
                value_dict = functions.process_csv_file(csv_file_path)
                hospital_frequency_cipher_dict[value] = value_dict

        hospital_query_number_dict_cipher = {key: 0 for key in functions.data}
        for inner_dict in hospital_frequency_cipher_dict.values():
            for key, value in inner_dict.items():
                hospital_query_number_dict_cipher[key] += value

        hospital_query_number_cipher = list(hospital_query_number_dict_cipher.values())
        hospital_query_freqencry_matrix_cipher = []
        for key1, inner_dict in hospital_frequency_cipher_dict.items():
            row = []
            for idx, key2 in enumerate(inner_dict.keys()):
                if hospital_query_number_cipher[idx] != 0:
                    row.append(inner_dict[key2] / hospital_query_number_cipher[idx])
                else:
                    row.append(0)
            hospital_query_freqencry_matrix_cipher.append(row)

        diagnosis_cipher = functions.extract_columns(matrix_cipher, selected_column_sse[1])
        diagnosis_cipher_list = list(set(diagnosis_cipher))

        diagnosis_frequency_cipher_dict = {}

        for value in diagnosis_cipher_list:
            csv_file_path = os.path.join(frequency_folder, f'{value}.csv')
            if os.path.exists(csv_file_path):
                value_dict = functions.process_csv_file(csv_file_path)
                diagnosis_frequency_cipher_dict[value] = value_dict

        diagnosis_query_number_dict_cipher = {key: 0 for key in functions.data}
        for inner_dict in diagnosis_frequency_cipher_dict.values():
            for key, value in inner_dict.items():
                diagnosis_query_number_dict_cipher[key] += value

        diagnosis_query_number_cipher = list(diagnosis_query_number_dict_cipher.values())
        diagnosis_query_freqencry_matrix_cipher = []
        for key1, inner_dict in diagnosis_frequency_cipher_dict.items():
            row = []
            for idx, key2 in enumerate(inner_dict.keys()):
                if diagnosis_query_number_cipher[idx] != 0:
                    row.append(inner_dict[key2] / diagnosis_query_number_cipher[idx])
                else:
                    row.append(0)
            diagnosis_query_freqencry_matrix_cipher.append(row)
        
        hospital_plain = functions.extract_columns(matrix_plain, selected_column_sse[0])
        hospital_plain_list = list(set(hospital_plain))

        empty_dict={'2009.01-2009.12': 0,
        '2010.01-2010.12': 0,
        '2011.01-2011.12': 0,
        '2012.01-2012.12': 0,
        '2013.01-2013.12': 0,
        '2014.01-2014.12': 0,
        '2015.01-2015.12': 0,
        '2016.01-2016.12': 0,
        '2017.01-2017.12': 0,
        '2018.01-2018.12': 0,
        '2019.01-2019.12': 0,
        '2020.01-2020.12': 0,
        '2021.01-2021.12': 0,
        '2022.01-2022.12': 0,
        '2023.01-2023.12': 0,
        '2024.01-2024.12': 0}
        hospital_frequency_plain_dict = {}
        for value in hospital_plain_list:
            csv_file_path = os.path.join(frequency_folder, f'{value}.csv')
            if os.path.exists(csv_file_path):
                value_dict = functions.process_csv_file(csv_file_path)
                hospital_frequency_plain_dict[value] = value_dict
            else:
                hospital_frequency_plain_dict[value] = empty_dict

        hospital_query_number_dict_plain = {key: 0 for key in functions.data}
        for inner_dict in hospital_frequency_plain_dict.values():
            for key, value in inner_dict.items():
                hospital_query_number_dict_plain[key] += value

        hospital_query_number_plain = list(hospital_query_number_dict_plain.values())
        hospital_query_freqencry_matrix_plain = []
        for key1, inner_dict in hospital_frequency_plain_dict.items():
            row = []
            for idx, key2 in enumerate(inner_dict.keys()):
                if hospital_query_number_plain[idx] != 0:
                    row.append(inner_dict[key2] / hospital_query_number_plain[idx])
                else:
                    row.append(0)
            hospital_query_freqencry_matrix_plain.append(row)

        Nd = len(matrix_cipher_sse[1:])

        diagnosis_plain = functions.extract_columns(matrix_plain, selected_column_sse[1])
        diagnosis_plain_list = list(set(diagnosis_plain))

        diagnosis_frequency_plain_dict = {}

        for value in diagnosis_plain_list:
            csv_file_path = os.path.join(frequency_folder, f'{value}.csv')
            if os.path.exists(csv_file_path):
                value_dict = functions.process_csv_file(csv_file_path)
                diagnosis_frequency_plain_dict[value] = value_dict
            else:
                diagnosis_frequency_plain_dict[value] = empty_dict

        diagnosis_query_number_dict_plain = {key: 0 for key in functions.data}
        for inner_dict in diagnosis_frequency_plain_dict.values():
            for key, value in inner_dict.items():
                diagnosis_query_number_dict_plain[key] += value

        diagnosis_query_number_plain = list(diagnosis_query_number_dict_plain.values())
        diagnosis_query_freqencry_matrix_plain = []
        for key1, inner_dict in diagnosis_frequency_plain_dict.items():
            row = []
            for idx, key2 in enumerate(inner_dict.keys()):
                if diagnosis_query_number_plain[idx] != 0:
                    row.append(inner_dict[key2] / diagnosis_query_number_plain[idx])
                else:
                    row.append(0)
            diagnosis_query_freqencry_matrix_plain.append(row)

        hospital_volumn_cipher_sse = []
        for word in hospital_cipher_list:
            count = sum([1 for row in matrix_cipher_sse[1:] if word.lower() in [cell.lower() for cell in row]]) / len(matrix_cipher_sse[1:])
            hospital_volumn_cipher_sse.append(count)

        hospital_volumn_plain_sse = []
        for word in hospital_plain_list:
            count = sum([1 for row in matrix_plain_sse[1:] if word.lower() in [cell.lower() for cell in row]]) / len(matrix_plain_sse[1:])
            hospital_volumn_plain_sse.append(count)

        diagnosis_volumn_cipher_sse = []
        for word in diagnosis_cipher_list:
            count = sum([1 for row in matrix_cipher_sse[1:] if word.lower() in [cell.lower() for cell in row]]) / len(matrix_cipher_sse[1:])
            diagnosis_volumn_cipher_sse.append(count)

        diagnosis_volumn_plain_sse = []
        for word in diagnosis_plain_list:
            count = sum([1 for row in matrix_plain_sse[1:] if word.lower() in [cell.lower() for cell in row]]) / len(matrix_plain_sse[1:])
            diagnosis_volumn_plain_sse.append(count)

        C_f_hospital = []
        for i_hospital in range(len(hospital_query_freqencry_matrix_plain)):
            Cf_i = []
            for j_hospital in range(len(hospital_query_freqencry_matrix_cipher)):
                Cf_ij = 0
                for k_hospital in range(len(hospital_query_number_cipher)):
                    f_cipher_jk = hospital_query_freqencry_matrix_cipher[j_hospital][k_hospital]
                    f_plain_ik = round(hospital_query_freqencry_matrix_plain[i_hospital][k_hospital], 6)
                    eta_k = hospital_query_number_cipher[k_hospital]

                    if f_plain_ik == 0:
                        Cf_ij += 0
                    else:
                        v_ij = -eta_k * f_cipher_jk * np.log(f_plain_ik)
                        Cf_ij += v_ij.item()
                
                Cf_i.append(Cf_ij)
            C_f_hospital.append(Cf_i)

        C_f_diagnosis = []
        for i_diagnosis in range(len(diagnosis_query_freqencry_matrix_plain)):
            Cf_i = []
            for j_diagnosis in range(len(diagnosis_query_freqencry_matrix_cipher)):
                Cf_ij = 0
                for k_diagnosis in range(len(diagnosis_query_number_cipher)):
                    f_cipher_jk = diagnosis_query_freqencry_matrix_cipher[j_diagnosis][k_diagnosis]
                    f_plain_ik = round(diagnosis_query_freqencry_matrix_plain[i_diagnosis][k_diagnosis], 6)
                    eta_k = diagnosis_query_number_cipher[k_diagnosis]

                    if f_plain_ik == 0:
                        Cf_ij += 0
                    else:
                        v_ij = -eta_k * f_cipher_jk * np.log(f_plain_ik)
                        Cf_ij += v_ij.item()
                
                Cf_i.append(Cf_ij)
            C_f_diagnosis.append(Cf_i)

        C_v_hospital = []
        for i_hospital in range(len(hospital_volumn_plain_sse)):
            Cv_i = []
            for j_hospital in range(len(hospital_volumn_cipher_sse)):
                Cv_ij = - (Nd * hospital_volumn_cipher_sse[j_hospital] * np.log(hospital_volumn_plain_sse[i_hospital]) + Nd * (1 - hospital_volumn_cipher_sse[j_hospital]) * np.log(1-hospital_volumn_plain_sse[i_hospital]))
                Cv_i.append(Cv_ij.item())
            C_v_hospital.append(Cv_i)

        C_v_diagnosis = []
        for i_diagnosis in range(len(diagnosis_volumn_plain_sse)):
            Cv_i = []
            for j_diagnosis in range(len(diagnosis_volumn_cipher_sse)):
                Cv_ij = - (Nd * diagnosis_volumn_cipher_sse[j_diagnosis] * np.log(diagnosis_volumn_plain_sse[i_diagnosis]) + Nd * (1 - diagnosis_volumn_cipher_sse[j_diagnosis]) * np.log(1-diagnosis_volumn_plain_sse[i_diagnosis]))
                Cv_i.append(Cv_ij.item())
            C_v_diagnosis.append(Cv_i)

        alpha = 0.005
        cost_matrix_hospital = [[alpha * C_f_hospital[i][j] + (1 - alpha) * C_v_hospital[i][j] for j in range(len(C_f_hospital[0]))] for i in range(len(C_f_hospital))]
        row_ind_hospital, col_ind_hospital = hungarian(cost_matrix_hospital)

        cost_matrix_diagnosis = [[alpha * C_f_diagnosis[i][j] + (1 - alpha) * C_v_diagnosis[i][j] for j in range(len(C_f_diagnosis[0]))] for i in range(len(C_f_diagnosis))]
        row_ind_diagnosis, col_ind_diagnosis = hungarian(cost_matrix_diagnosis)

        pred_dict_ = {}
        for idx_hospital in range(min(len(row_ind_hospital), len(col_ind_hospital))):
            cipher_idx = row_ind_hospital[idx_hospital] - 1
            plain_idx = col_ind_hospital[idx_hospital] - 1

            if cipher_idx < len(hospital_cipher_list) and plain_idx < len(hospital_plain_list):
                pred_dict_[hospital_cipher_list[cipher_idx]] = hospital_plain_list[plain_idx]

        for idx_diagnosis in range(min(len(row_ind_diagnosis), len(col_ind_diagnosis))):
            cipher_idx = row_ind_diagnosis[idx_diagnosis] - 1
            plain_idx = col_ind_diagnosis[idx_diagnosis] - 1

            if cipher_idx < len(diagnosis_cipher_list) and plain_idx < len(diagnosis_plain_list):
                pred_dict_[diagnosis_cipher_list[cipher_idx]] = diagnosis_plain_list[plain_idx]

        count = 0
        for key,value in pred_dict_.items():
            if key == value:
                value_mapping_od[key] = value
                count += 1
        print(count)
        print(len(value_mapping_od))
        keyword_number = functions.count_keywords(functions.generate_submatrix(matrix_cipher, selected_columns[1:])[1:])
        recovered_keywords.append(len(value_mapping_od))
        keywords_count.append(keyword_number)
        
    avg_keyword_number = sum(keywords_count) / 5
    avg_recovered_keywords = sum(recovered_keywords) / 5

    new_file_name = "text_" + str(i) + ".txt"
    outputPath = os.path.join(out,new_file_name)
    print(outputPath)
    with open(outputPath,"w") as f:
        f.write("keywords number: " + str(avg_keyword_number) + " successfully recovered keyword number: " + str(avg_recovered_keywords))

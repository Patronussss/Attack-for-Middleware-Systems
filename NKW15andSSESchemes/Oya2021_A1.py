import sys
sys.path.append("/CCS2026/") 
import functions
from FrequencyAnalysisAttack import FrequencyAnalysisAttack
import pandas as pd
import numpy as np
from collections import Counter
import time
import os
import networkx as nx
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment as hungarian
from AttributeRecoverAttack import AttributeRecoverAttack

root = "dataset/text_508029.csv"
base = [187500]
matrix = functions.read_csv_to_matrix(root)
frac_list = [0.9]

for ind in base:
    for frac in frac_list:
        for time in range(1,51):
            out = 'result/A3/Oya21/' + str(frac) + '/' + str(time) + '/'
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

            print("------sse-----")
            matrix_cipher_sse = functions.generate_submatrix(matrix_cipher, selected_column_sse)
            matrix_plain_sse = functions.generate_submatrix(matrix_plain, selected_column_sse)

            hospital_cipher_col = None
            diagnosis_cipher_col = None
            for c, p in attr_mapping.items():
                if p == 'Hospital':
                    hospital_cipher_col = c
                elif p == 'Pincipal Diagnosis':
                    diagnosis_cipher_col = c

            hospital_cipher = functions.extract_columns(matrix_cipher, hospital_cipher_col) if hospital_cipher_col else functions.extract_columns(matrix_cipher, 'Hospital')
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

            diagnosis_cipher = functions.extract_columns(matrix_cipher, diagnosis_cipher_col) if diagnosis_cipher_col else functions.extract_columns(matrix_cipher, 'Pincipal Diagnosis')
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
            
            hospital_plain = functions.extract_columns(matrix_plain, hospital_cipher_col) if hospital_cipher_col else functions.extract_columns(matrix_plain, 'Hospital')
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

            diagnosis_plain = functions.extract_columns(matrix_plain, diagnosis_cipher_col) if diagnosis_cipher_col else functions.extract_columns(matrix_plain, 'Pincipal Diagnosis')
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
            for i in range(len(hospital_query_freqencry_matrix_plain)):
                Cf_i = []
                for j in range(len(hospital_query_freqencry_matrix_cipher)):
                    Cf_ij = 0
                    for k in range(len(hospital_query_number_cipher)):
                        f_cipher_jk = hospital_query_freqencry_matrix_cipher[j][k]
                        f_plain_ik = round(hospital_query_freqencry_matrix_plain[i][k], 6)
                        eta_k = hospital_query_number_cipher[k]

                        if f_plain_ik == 0:
                            Cf_ij += 0
                        else:
                            v_ij = -eta_k * f_cipher_jk * np.log(f_plain_ik)
                            Cf_ij += v_ij.item()
                        
                    Cf_i.append(Cf_ij)
                C_f_hospital.append(Cf_i)

            C_f_diagnosis = []
            for i in range(len(diagnosis_query_freqencry_matrix_plain)):
                Cf_i = []
                for j in range(len(diagnosis_query_freqencry_matrix_cipher)):
                    Cf_ij = 0
                    for k in range(len(diagnosis_query_number_cipher)):
                        f_cipher_jk = diagnosis_query_freqencry_matrix_cipher[j][k]
                        f_plain_ik = round(diagnosis_query_freqencry_matrix_plain[i][k], 6)
                        eta_k = diagnosis_query_number_cipher[k]
                        if f_plain_ik == 0:
                            Cf_ij += 0
                        else:
                            v_ij = -eta_k * f_cipher_jk * np.log(f_plain_ik)
                            Cf_ij += v_ij.item()
                        
                    Cf_i.append(Cf_ij)
                C_f_diagnosis.append(Cf_i)

            C_v_hospital = []
            for i in range(len(hospital_volumn_plain_sse)):
                Cv_i = []
                for j in range(len(hospital_volumn_cipher_sse)):
                    Cv_ij = - (Nd * hospital_volumn_cipher_sse[j] * np.log(hospital_volumn_plain_sse[i]) + Nd * (1 - hospital_volumn_cipher_sse[j]) * np.log(1-hospital_volumn_plain_sse[i]))
                    Cv_i.append(Cv_ij.item())
                C_v_hospital.append(Cv_i)

            C_v_diagnosis = []
            for i in range(len(diagnosis_volumn_plain_sse)):
                Cv_i = []
                for j in range(len(diagnosis_volumn_cipher_sse)):
                    Cv_ij = - (Nd * diagnosis_volumn_cipher_sse[j] * np.log(diagnosis_volumn_plain_sse[i]) + Nd * (1 - diagnosis_volumn_cipher_sse[j]) * np.log(1-diagnosis_volumn_plain_sse[i]))
                    Cv_i.append(Cv_ij.item())
                C_v_diagnosis.append(Cv_i)

            alpha = 0.005
            cost_matrix_hospital = [[alpha * C_f_hospital[i][j] + (1 - alpha) * C_v_hospital[i][j] for j in range(len(C_f_hospital[0]))] for i in range(len(C_f_hospital))]
            row_ind_hospital, col_ind_hospital = hungarian(cost_matrix_hospital)

            cost_matrix_diagnosis = [[alpha * C_f_diagnosis[i][j] + (1 - alpha) * C_v_diagnosis[i][j] for j in range(len(C_f_diagnosis[0]))] for i in range(len(C_f_diagnosis))]
            row_ind_diagnosis, col_ind_diagnosis = hungarian(cost_matrix_diagnosis)

            pred_dict_ = {}
            for i in range(min(len(row_ind_hospital), len(col_ind_hospital))):
                cipher_idx = row_ind_hospital[i] - 1
                plain_idx = col_ind_hospital[i] - 1

                if cipher_idx < len(hospital_cipher_list) and plain_idx < len(hospital_plain_list):
                    pred_dict_[hospital_cipher_list[cipher_idx]] = hospital_plain_list[plain_idx]

            for i in range(min(len(row_ind_diagnosis), len(col_ind_diagnosis))):
                cipher_idx = row_ind_diagnosis[i] - 1
                plain_idx = col_ind_diagnosis[i] - 1

                if cipher_idx < len(diagnosis_cipher_list) and plain_idx < len(diagnosis_plain_list):
                    pred_dict_[diagnosis_cipher_list[cipher_idx]] = diagnosis_plain_list[plain_idx]

            count = 0
            for key,value in pred_dict_.items():
                if key == value:
                    value_mapping_od[key] = value
                    count += 1
            print(count)
            print(len(value_mapping_od))
            keyword_number = functions.count_keywords(functions.generate_submatrix(matrix_cipher, selected_columns_noid)[1:])
            
            kw_c = keyword_number
            kw_s_c = len(value_mapping_od)

            new_file_name = "text_" + str(ind) + ".txt"
            outputPath = os.path.join(out,new_file_name)
            print(outputPath)
            with open(outputPath,"w") as f:
                f.write(" keywords number: " + str(kw_c) + " successfully recovered keyword number: " + str(kw_s_c) + "percent of file" + str(frac))

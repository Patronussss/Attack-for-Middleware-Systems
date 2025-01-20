import csv
from tqdm import tqdm
import random
from collections import Counter
import functions
import numpy as np
import os
from collections import defaultdict
import math
from scipy.optimize import linear_sum_assignment as hungarian

index_list = [1,2,3,4]

for il in index_list:
    base = 'F:/Desktop/text/'+ str(il)+ 'q2010/'
    # target = '/Users/cherry/Desktop/zengli/'+ str(il)+ 'q2010plain/'  
    out = 'F:/Desktop/text/new/Oya2021_new/'+ str(il)+ 'q2010 output of Oya2021/'
    lists = [i[:-4] for i in os.listdir(out)]
    file_list = os.listdir(base)
    for file_name in sorted(file_list):
        if file_name[:-4] not in lists:
            filePath = os.path.join(base,file_name)
            matrix = functions.read_csv_to_matrix(filePath)
            # filePathPlain = "D:\\Users\\Lizi\\Desktop\\text\\2q2010\\text_450950.csv"
            filePathPlain = "2015/PUDF_base1_1q2015.csv"
            matrixP = functions.read_csv_to_matrix(filePathPlain)

            selected_columns = ['Record ID', 'Age', 'Admission Type', 'Length of stay', 'Risk', 'Gender', 'Race','Hospital', 'Pincipal Diagnosis']
            selected_columns_noid = ['Age', 'Admission Type', 'Length of stay', 'Risk', 'Gender', 'Race','Hospital', 'Pincipal Diagnosis']

            matrix_cipher = functions.generate_submatrix(matrix, selected_columns)
            matrix_plain = functions.generate_submatrix(matrixP, selected_columns)

            # 针对OPE的攻击，首先判断每一列的数据种类，如果|C| = |M|那么可以直接排序
            selected_column_ope = ['Record ID', 'Age', 'Admission Type', 'Length of stay', 'Risk']
            matrix_cipher_ope = functions.generate_submatrix(matrix_cipher, selected_column_ope)
            matrix_plain_ope = functions.generate_submatrix(matrix_plain, selected_column_ope)
            value_mapping = {}
            # Age
            column_cipher_age = functions.extract_columns(matrix_cipher_ope, 'Age')
            age_cipher_count = Counter(column_cipher_age)
            keyword_count_cipher_age = len(age_cipher_count)

            column_plain_age = functions.extract_columns(matrix_plain_ope, 'Age')
            age_plain_count = Counter(column_plain_age)
            keyword_count_plain_age = len(age_plain_count)
            value_mapping = {}

            if keyword_count_cipher_age == keyword_count_plain_age:
                sorted_cipher_age = sorted(set(column_cipher_age))
                sorted_plain_age = sorted(set(column_plain_age))
                mapping_ope_age = {key: value for key, value in zip(sorted_cipher_age, sorted_plain_age)}
                value_mapping.update(mapping_ope_age)
            else:
                mapping_ope_age = functions.find_optimal_mapping(column_cipher_age, column_plain_age)
                value_mapping.update(mapping_ope_age)

            # Admission Type
            column_cipher_admi = functions.extract_columns(matrix_cipher_ope, 'Admission Type')
            admi_cipher_count = Counter(column_cipher_admi)
            keyword_count_cipher_admi = len(admi_cipher_count)

            column_plain_admi = functions.extract_columns(matrix_plain_ope, 'Admission Type')
            admi_plain_count = Counter(column_plain_admi)
            keyword_count_plain_admi = len(admi_plain_count)

            if keyword_count_cipher_admi == keyword_count_plain_admi:
                sorted_cipher_admi = sorted(set(column_cipher_admi))
                sorted_plain_admi = sorted(set(column_plain_admi))
                mapping_ope_admi = {key: value for key, value in zip(sorted_cipher_admi, sorted_plain_admi)}
                value_mapping.update(mapping_ope_admi)
            else:
                mapping_ope_admi = functions.find_optimal_mapping(column_cipher_admi, column_plain_admi)
                value_mapping.update(mapping_ope_admi)

            # Risk
            column_cipher_risk = functions.extract_columns(matrix_cipher_ope, 'Risk')
            risk_cipher_count = Counter(column_cipher_risk)
            keyword_count_cipher_risk = len(risk_cipher_count)

            column_plain_risk = functions.extract_columns(matrix_plain_ope, 'Risk')
            risk_plain_count = Counter(column_plain_risk)
            keyword_count_plain_risk = len(risk_plain_count)

            if keyword_count_cipher_risk == keyword_count_plain_risk:
                sorted_cipher_risk = sorted(set(column_cipher_risk))
                sorted_plain_risk = sorted(set(column_plain_risk))
                mapping_ope_risk = {key: value for key, value in zip(sorted_cipher_risk, sorted_plain_risk)}
                value_mapping.update(mapping_ope_risk)
            else:
                mapping_ope_risk = functions.find_optimal_mapping(column_cipher_risk, column_plain_risk)
                value_mapping.update(mapping_ope_risk)

            # Length of Stay
            column_cipher_stay = functions.extract_columns(matrix_cipher_ope, 'Length of stay')
            stay_cipher_count = Counter(column_cipher_stay)
            keyword_count_cipher_stay = len(stay_cipher_count)

            column_plain_stay = functions.extract_columns(matrix_plain_ope, 'Length of stay')
            stay_plain_count = Counter(column_plain_stay)
            keyword_count_plain_stay = len(stay_plain_count)

            if keyword_count_cipher_stay == keyword_count_plain_stay:
                sorted_cipher_stay = sorted(set(column_cipher_stay))
                sorted_plain_stay = sorted(set(column_plain_stay))
                mapping_ope_stay = {key: value for key, value in zip(sorted_cipher_stay, sorted_plain_stay)}
                value_mapping.update(mapping_ope_stay)
            else:
                sorted_cipher_stay = sorted(column_cipher_stay, key=functions.custom_sort)
                sorted_plain_stay = sorted(column_plain_stay, key=functions.custom_sort)
                mapping_ope_stay = functions.find_optimal_mapping(sorted_cipher_stay, sorted_plain_stay)
                value_mapping.update(mapping_ope_stay)

            keyword_count_stay = functions.count_keywords(functions.generate_submatrix(matrix_cipher,[
                'Length of stay'
                # ,'Age', 'Admission Type', 'Risk'
                # ,'Discharge', 'Gender', 'Race'
                ])[1:]) 

            keyword_count_aar = functions.count_keywords(functions.generate_submatrix(matrix_cipher,[
                # 'Length of stay',
                'Age', 'Admission Type', 'Risk'
                # ,'Discharge', 'Gender', 'Race'
                ])[1:]) 
            print(f"length of stay: {keyword_count_stay}")
            print(f"Age + Admission Type + Risk: {keyword_count_aar}")

            # DET的部分，计算频率和种类
            selected_columns_det = ['Gender', 'Race']
            matrix_cipher_det = functions.generate_submatrix(matrix_cipher, selected_columns_det)
            matrix_plain_det = functions.generate_submatrix(matrix_plain, selected_columns_det)

            # 计算频率
            element_cipher_det = functions.column_frequencies(matrix_cipher_det[1:])
            element_plain_det = functions.column_frequencies(matrix_plain_det[1:])

            result = {}
            for column, dict1 in element_cipher_det.items():
                dict2 = element_plain_det[column]
                temp = functions.find_closest_mapping(dict1,dict2)
                result.update(temp)
            print(f"DET的关键字个数为：{len(result)}")

            for key, dict1 in element_cipher_det.items():
                dict2 = element_plain_det[key]
                temp = functions.find_closest_mapping(dict1, dict2)
                value_mapping.update(temp)
            
            value_mapping_od = {}
            for key, value in value_mapping.items():
                if key == value:
                    value_mapping_od[key] = value

            selected_columns_od = ['Record ID', 'Age', 'Admission Type', 'Length of stay', 'Risk','Gender', 'Race']

            matrix_cipher_od = functions.generate_submatrix(matrix_cipher, selected_columns_od)  # 经过OPE和DET后恢复了多少个记录
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
                # index_cipher = recordid_cipher[id_c]
                row_cipher = matrix_cipher[recordid_cipher[id_c]]
                index_plain = recordid_plain[id_p]
                row_plain = matrix_plain[index_plain]
                for i in range(1,len(row_cipher)):
                    value_mapping[row_cipher[i]] = row_plain[i]

            for key, value in value_mapping.items():
                if key == value:
                    value_mapping_od[key] = value

            # SSE
            print("------sse-----")
            selected_column_sse = ['Hospital','Pincipal Diagnosis']
            matrix_cipher_sse = functions.generate_submatrix(matrix_cipher, selected_column_sse)
            matrix_plain_sse = functions.generate_submatrix(matrix_plain, selected_column_sse)
            # Query Frequency hospital 
            hospital_cipher = functions.extract_columns(matrix_cipher, 'Hospital')
            hospital_cipher_list = list(set(hospital_cipher))
            # cipher_list = list(set(hospital_cipher_list + diagnosis_cipher_list))

            hospital_frequency_cipher_dict = {}

            frequency_folder = 'frequency/'
            for value in hospital_cipher_list:
                csv_file_path = os.path.join(frequency_folder, f'{value}.csv')
                if os.path.exists(csv_file_path):
                    value_dict = functions.process_csv_file(csv_file_path)
                    hospital_frequency_cipher_dict[value] = value_dict
            # 初始化一个与字典中value字典相同key的字典，用于存储加和结果
            hospital_query_number_dict_cipher = {key: 0 for key in functions.data}
            # 计算每个key对应值的加和
            for inner_dict in hospital_frequency_cipher_dict.values():
                for key, value in inner_dict.items():
                    hospital_query_number_dict_cipher[key] += value

            # 将结果转换为向量
            hospital_query_number_cipher = list(hospital_query_number_dict_cipher.values())
            hospital_query_freqencry_matrix_cipher = []
            # 计算矩阵，每一行代表D的一个key1，每一列代表dict中的key2
            for key1, inner_dict in hospital_frequency_cipher_dict.items():
                row = []
                for idx, key2 in enumerate(inner_dict.keys()):
                    # 检查除数是否为零
                    if hospital_query_number_cipher[idx] != 0:
                        row.append(inner_dict[key2] / hospital_query_number_cipher[idx])
                    else:
                        row.append(0)  # 或者根据实际需求设置为其他值
                hospital_query_freqencry_matrix_cipher.append(row)

            # Query Frequency diagnosis
            diagnosis_cipher = functions.extract_columns(matrix_cipher, 'Pincipal Diagnosis')
            diagnosis_cipher_list = list(set(diagnosis_cipher))

            diagnosis_frequency_cipher_dict = {}

            for value in diagnosis_cipher_list:
                csv_file_path = os.path.join(frequency_folder, f'{value}.csv')
                if os.path.exists(csv_file_path):
                    value_dict = functions.process_csv_file(csv_file_path)
                    diagnosis_frequency_cipher_dict[value] = value_dict

            # 初始化一个与字典中value字典相同key的字典，用于存储加和结果
            diagnosis_query_number_dict_cipher = {key: 0 for key in functions.data}
            # 计算每个key对应值的加和
            for inner_dict in diagnosis_frequency_cipher_dict.values():
                for key, value in inner_dict.items():
                    diagnosis_query_number_dict_cipher[key] += value

            # 将结果转换为向量
            diagnosis_query_number_cipher = list(diagnosis_query_number_dict_cipher.values())
            diagnosis_query_freqencry_matrix_cipher = []
            # 计算矩阵，每一行代表D的一个key1，每一列代表dict中的key2
            for key1, inner_dict in diagnosis_frequency_cipher_dict.items():
                row = []
                for idx, key2 in enumerate(inner_dict.keys()):
                    # 检查除数是否为零
                    if diagnosis_query_number_cipher[idx] != 0:
                        row.append(inner_dict[key2] / diagnosis_query_number_cipher[idx])
                    else:
                        row.append(0)  # 或者根据实际需求设置为其他值
                diagnosis_query_freqencry_matrix_cipher.append(row) 
            
            # Query Frequency_plain hospital
            hospital_plain = functions.extract_columns(matrix_plain, 'Hospital')
            hospital_plain_list = list(set(hospital_plain))

            # plain_list = list(set(hospital_plain_list + diagnosis_plain_list))

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
            # frequency_folder = 'frequency/'
            for value in hospital_plain_list:
                csv_file_path = os.path.join(frequency_folder, f'{value}.csv')
                if os.path.exists(csv_file_path):
                    value_dict = functions.process_csv_file(csv_file_path)
                    hospital_frequency_plain_dict[value] = value_dict
                else:
                    hospital_frequency_plain_dict[value] = empty_dict

            # for value in diagnosis_plain_list:
            #     csv_file_path = os.path.join(frequency_folder, f'{value}.csv')
            #     if os.path.exists(csv_file_path):
            #         value_dict = functions.process_csv_file(csv_file_path)
            #         frequency_plain_dict[value] = value_dict

            # 初始化一个与字典中value字典相同key的字典，用于存储加和结果
            hospital_query_number_dict_plain = {key: 0 for key in functions.data}
            # 计算每个key对应值的加和
            for inner_dict in hospital_frequency_plain_dict.values():
                for key, value in inner_dict.items():
                    hospital_query_number_dict_plain[key] += value

            # 将结果转换为向量
            hospital_query_number_plain = list(hospital_query_number_dict_plain.values())
            hospital_query_freqencry_matrix_plain = []
            # 计算矩阵，每一行代表D的一个key1，每一列代表dict中的key2
            for key1, inner_dict in hospital_frequency_plain_dict.items():
                row = []
                for idx, key2 in enumerate(inner_dict.keys()):
                    # 检查除数是否为零
                    if hospital_query_number_plain[idx] != 0:
                        row.append(inner_dict[key2] / hospital_query_number_plain[idx])
                    else:
                        row.append(0)  # 或者根据实际需求设置为其他值
                hospital_query_freqencry_matrix_plain.append(row)

            # 记录的总数
            Nd = len(matrix_cipher_sse[1:])

            # Query Frequency_plain diagnosis
            diagnosis_plain = functions.extract_columns(matrix_plain, 'Pincipal Diagnosis')
            diagnosis_plain_list = list(set(diagnosis_plain))

            diagnosis_frequency_plain_dict = {}

            for value in diagnosis_plain_list:
                csv_file_path = os.path.join(frequency_folder, f'{value}.csv')
                if os.path.exists(csv_file_path):
                    value_dict = functions.process_csv_file(csv_file_path)
                    diagnosis_frequency_plain_dict[value] = value_dict
                else:
                    diagnosis_frequency_plain_dict[value] = empty_dict

            # 初始化一个与字典中value字典相同key的字典，用于存储加和结果
            diagnosis_query_number_dict_plain = {key: 0 for key in functions.data}
            # 计算每个key对应值的加和
            for inner_dict in diagnosis_frequency_plain_dict.values():
                for key, value in inner_dict.items():
                    diagnosis_query_number_dict_plain[key] += value

            # 将结果转换为向量
            diagnosis_query_number_plain = list(diagnosis_query_number_dict_plain.values())
            diagnosis_query_freqencry_matrix_plain = []
            # 计算矩阵，每一行代表D的一个key1，每一列代表dict中的key2
            for key1, inner_dict in diagnosis_frequency_plain_dict.items():
                row = []
                for idx, key2 in enumerate(inner_dict.keys()):
                    # 检查除数是否为零
                    if diagnosis_query_number_plain[idx] != 0:
                        row.append(inner_dict[key2] / diagnosis_query_number_plain[idx])
                    else:
                        row.append(0)  # 或者根据实际需求设置为其他值
                diagnosis_query_freqencry_matrix_plain.append(row)

            # volumn hospital
            hospital_volumn_cipher_sse = []

            for word in hospital_cipher_list:
                count = sum([1 for row in matrix_cipher_sse[1:] if word.lower() in [cell.lower() for cell in row]]) / len(matrix_cipher_sse[1:])
                hospital_volumn_cipher_sse.append(count)

            hospital_volumn_plain_sse = []

            for word in hospital_plain_list:
                count = sum([1 for row in matrix_plain_sse[1:] if word.lower() in [cell.lower() for cell in row]]) / len(matrix_plain_sse[1:])
                hospital_volumn_plain_sse.append(count)

            # volumn diagnosis
            diagnosis_volumn_cipher_sse = []

            for word in diagnosis_cipher_list:
                count = sum([1 for row in matrix_cipher_sse[1:] if word.lower() in [cell.lower() for cell in row]]) / len(matrix_cipher_sse[1:])
                diagnosis_volumn_cipher_sse.append(count)


            diagnosis_volumn_plain_sse = []

            for word in diagnosis_plain_list:
                count = sum([1 for row in matrix_plain_sse[1:] if word.lower() in [cell.lower() for cell in row]]) / len(matrix_plain_sse[1:])
                diagnosis_volumn_plain_sse.append(count)

            # C_f hospital
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

            # C_f diagnosis
            C_f_diagnosis = []

            for i in range(len(diagnosis_query_freqencry_matrix_plain)):
                Cf_i = []
                for j in range(len(diagnosis_query_freqencry_matrix_cipher)):
                    Cf_ij = 0
                    for k in range(len(diagnosis_query_number_cipher)):
                        f_cipher_jk = diagnosis_query_freqencry_matrix_cipher[j][k]
                        f_plain_ik = round(diagnosis_query_freqencry_matrix_plain[i][k], 6)
                        eta_k = diagnosis_query_number_cipher[k]
                        # print(f_plain_ik)
                        if f_plain_ik == 0:
                            Cf_ij += 0
                        else:
                            v_ij = -eta_k * f_cipher_jk * np.log(f_plain_ik)
                            Cf_ij += v_ij.item()
                        
                    Cf_i.append(Cf_ij)
                C_f_diagnosis.append(Cf_i)

            # # C_v hospital
            C_v_hospital = []

            for i in range(len(hospital_volumn_plain_sse)):
                Cv_i = []
                for j in range(len(hospital_volumn_cipher_sse)):
                    Cv_ij = - (Nd * hospital_volumn_cipher_sse[j] * np.log(hospital_volumn_plain_sse[i]) + Nd * (1 - hospital_volumn_cipher_sse[j]) * np.log(1-hospital_volumn_plain_sse[i]))
                    Cv_i.append(Cv_ij.item())
                C_v_hospital.append(Cv_i)

            # # C_v diagnosis
            C_v_diagnosis = []

            for i in range(len(diagnosis_volumn_plain_sse)):
                Cv_i = []
                for j in range(len(diagnosis_volumn_cipher_sse)):
                    Cv_ij = - (Nd * diagnosis_volumn_cipher_sse[j] * np.log(diagnosis_volumn_plain_sse[i]) + Nd * (1 - diagnosis_volumn_cipher_sse[j]) * np.log(1-diagnosis_volumn_plain_sse[i]))
                    Cv_i.append(Cv_ij.item())
                C_v_diagnosis.append(Cv_i)

            # 匈牙利算法 矩阵的某个值等于1，说明该行所代表的密文对应的明文是该列所代表的关键字
            alpha = 0.005
            # cost_matrix = C_f * alpha + C_v * (1 - alpha)
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

            file_extension = os.path.splitext(file_name)[1]
            new_file_name = file_name.replace(file_extension, ".txt")
            outputPath = os.path.join(out,new_file_name)
            print(outputPath)
            with open(outputPath,"w") as f:
                f.write("keywords number: " + str(keyword_number) + " successfully recovered keyword number: " + str(len(value_mapping_od)))


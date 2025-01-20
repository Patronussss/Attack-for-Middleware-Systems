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
import functions

# def find_closest_mapping(dict1, dict2):
#     mapping = {}
#     for key1, value1 in dict1.items():
#         min_distance = np.inf
#         candidate = key1
#         for key2, value2 in dict2.items():
#             distance = abs(value1 - value2)
#             if distance < min_distance:
#                 min_distance = distance
#                 candidate = key2
#         mapping[key1] = candidate
#     return mapping

# def replace_values_with_none(matrix, reference_dict):
#     # 遍历每一行，从第二列开始检查
#     for row in matrix:
#         for i in range(1, len(row)):
#             if row[i] not in reference_dict:
#                 row[i] = "zlzlzl"
#     return matrix
# # 读取CSV文件并生成矩阵
# def read_csv_to_matrix(file_path):
#     matrix = []
#     with open(file_path, 'r') as file:
#         reader = csv.reader(file)
#         for row in reader:
#             matrix.append(row)
#     return matrix

# def generate_submatrix(matrix, columns):
#     # 获取表头
#     header = matrix[0]
#     # 找到选定列的索引
#     header_indices = [header.index(col) for col in columns]
#     # 创建子矩阵并保留表头
#     submatrix = [columns]  # 添加新的表头
#     for row in matrix[1:]:
#         subrow = [row[i] for i in header_indices]
#         submatrix.append(subrow)
#     return submatrix

# # 定义一个函数，用于统计矩阵中出现的关键字数量
# def count_keywords(matrix):
#     # 创建一个空集合，用于存储所有出现的关键字
#     keywords = set()
#     # 遍历矩阵中的所有元素
#     for row in matrix:
#         for element in row:
#             # 使用正则表达式提取关键字
#             keywords.update(re.findall(r"\b[\w\s]+\b", element))
#     # 返回关键字的数量
#     return len(keywords)

# def custom_sort(value):
#     # 尝试将字符串转换为数字
#     try:
#         return int(value)  # 如果是纯数字，返回整数
#     except ValueError:
#         return value  # 否则返回原字符串

# def extract_columns(matrix, target_column):
#     # 找出目标列的索引
#     headers = matrix[0]
#     index = headers.index(target_column)
#     # 提取目标列的数据
#     return [row[index] for row in matrix[1:]]

# def all_numeric(input_list):
#     return all(item.isdigit() for item in input_list)

# def compute_frequency_and_cdf(data):
#     # if all_numeric(data):
#     #     data1 = [int(i) for i in data]
#     #     freq = Counter(data1)
#     # else:
#         # freq = Counter(data)
#     freq = Counter(data)
#     total = sum(freq.values())
#     cdf = {}
#     cumulative = 0

#     for key in sorted(freq):
#         count = freq[key]
#         cumulative += count
#         # freq[key] /= total
#         # cdf[key] = cumulative / total
#         freq[key] = (count / total) * 10000  # 将频率扩大10000倍
#         cdf[key] = (cumulative / total) * 10000  # 将CDF扩大10000倍

#     return freq, cdf

# def column_frequencies(matrix):
#     """
#     计算矩阵中每一列的元素出现频率，并将结果存储到字典中。
    
#     :param matrix: 2D numpy array or list of lists
#     :return: 字典，键为列索引，值为该列元素出现频率的字典
#     """
#     # 将输入数据转换为numpy数组
#     matrix = np.array(matrix)
#     # 初始化结果字典
#     frequencies = {}
    
#     # 遍历矩阵的每一列
#     for col_index in range(matrix.shape[1]):
#         col = matrix[:, col_index]  # 提取列数据
#         unique, counts = np.unique(col, return_counts=True)  # 计算唯一值及其计数
#         total = len(col)  # 该列的总元素数量
#         # 计算频率并保留5位小数
#         freq_dict = {str(key): round(count / total, 5) for key, count in zip(unique, counts)}
#         frequencies[col_index] = freq_dict  # 将频率字典存储到列索引对应的位置
    
#     return frequencies

# def find_optimal_mapping(data_c, data_z):
#     freq_c, cdf_c = compute_frequency_and_cdf(data_c)
#     freq_z, cdf_z = compute_frequency_and_cdf(data_z)

#     keys_c = sorted(freq_c.keys(), key=custom_sort)
#     keys_z = sorted(freq_z.keys(), key=custom_sort)

#     cost_matrix = np.zeros((len(keys_c), len(keys_z)))

#     for i, key_c in enumerate(keys_c):
#         for j, key_z in enumerate(keys_z):
#             # freq_diff = abs(freq_c[key_c] - freq_z[key_z])
#             # cdf_diff = abs(cdf_c[key_c] - cdf_z[key_z])
#             # sort_diff = ((i/len(key_c)-j/len(key_z))) ** 2 # /(len(key_c)+len(key_z))
#             freq_diff = (freq_c[key_c] - freq_z[key_z]) ** 2
#             cdf_diff = (cdf_c[key_c] - cdf_z[key_z]) ** 2
#             cost_matrix[i, j] = freq_diff + cdf_diff
#     row_ind, col_ind = linear_sum_assignment(cost_matrix)
#     mapping = {keys_c[i]: keys_z[j] for i, j in zip(row_ind, col_ind)}
#     return mapping

# def find_closest_mapping(dict1, dict2):
#     result = {}
#     for key1, value1 in dict1.items():
#         min_distance = np.inf
#         candidate = key1
#         for key2,value2 in dict2.items():
#             distance = abs(value1-value2)
#             if distance < min_distance:
#                 min_distance = distance
#                 candidate = key2
#         result[key1] = candidate
#     return result


# def find_unique_value(input_list):
#     value_counts = {}
    
#     for value in input_list:
#         if value in value_counts:
#             value_counts[value] += 1
#         else:
#             value_counts[value] = 1
    
#     unique_values = [key for key, count in value_counts.items() if count == 1]
#     return unique_values

# def create_rowid_dict(matrix, header, record_id_column):
#     rowid_dict = {}
#     record_id_index = header.index(record_id_column)
    
#     for index, row in enumerate(matrix[1:], start=1):
#         record_id = row[record_id_index]
#         rowid_dict[record_id] = index
    
#     return rowid_dict

# def count_frequency_multicol(matrix, column_names):
#     num_cols = matrix.shape[1]
#     frequency_dicts = [{} for _ in range(num_cols)]

#     for col_idx in range(num_cols):
#         column = matrix[:, col_idx]
#         total_count = len(column)

#         for key in column:
#             key_str = str(key)
#             frequency_dicts[col_idx][key_str] = frequency_dicts[col_idx].get(key_str, 0) + 1 / total_count

#     # Create the final dictionary with column names as keys
#     final_dict = {}

#     for col_idx, freq_dict in enumerate(frequency_dicts):
#         col_name = column_names[col_idx]
#         final_dict[col_name] = freq_dict

#     return final_dict

# def convert_to_date(year_month):
#     year, month = map(int, year_month.split('.'))
#     return datetime(year, month, 1)

# # 给定的数据列表
# data = ['2019.9-2019.12', '2020.1-2020.4', '2020.5-2020.8', '2020.9-2020.12',
#         '2021.1-2021.4', '2021.5-2021.8', '2021.9-2021.12',
#         '2022.1-2022.4', '2022.5-2022.8', '2022.9-2022.12',
#         '2023.1-2023.4', '2023.5-2023.8', '2023.9-2023.12',
#         '2024.1-2024.4', '2024.5-2024.8', '2024.9-2024.12']

# # 判断给定的 (year).(month) 在哪一个区间
# def find_period(year_month):
#     target_date = convert_to_date(year_month)
    
#     for idx, period in enumerate(data):
#         start_year_month, end_year_month = period.split('-')
#         start_date = convert_to_date(start_year_month)
#         end_date = convert_to_date(end_year_month)
        
#         if start_date <= target_date <= end_date:
#             return period

# # 读取CSV文件中的内容，生成字典
# def process_csv_file(file_path):
#     result_dict = {}
#     with open(file_path, 'r', newline='') as file:
#         reader = csv.reader(file)
#         next(reader)  # 跳过标题行
#         for row in reader:
#             time_period = row[0]
#             frequency = int(row[1])  # 假设频率为浮点数
#             times = time_period.split("-")
#             year = times[0]
#             month = times[1]
#             t = f'{year}.{month}'
#             key = find_period(t)
#             if key in result_dict:
#                 result_dict[key] += frequency
#             else:
#                 result_dict[key] = frequency
#     return result_dict

# def numToFreqency(data_dict):
#     # 计算每个子字典中时间段的频数总和
#     # try:
#     #     total_counts = {key: sum(sub_dict.values()) for key, sub_dict in data_dict.items()}
#     # except:
#     #     print(data_dict)
#     total_counts = {key: sum(sub_dict.values()) for key, sub_dict in data_dict.items()}
#     # 计算整个字典的总频数
#     total_frequency = sum(total_counts.values())
#     data_dict1 = {}
#     data_dict2 = {}
#     # 更新每个子字典中的值为频率
#     for key, sub_dict in data_dict.items():
#         sub_dict1 = {}
#         freq = 0
#         if not bool(sub_dict):
#             continue
#         else:
#             for period, count in sub_dict.items():
#                 sub_dict1[period] = count / total_frequency
#                 freq += count
#             data_dict1[key] = sub_dict1
#             data_dict2[key] = freq / total_frequency
#     return data_dict1, data_dict2


index_list = [1,2,3,4]

for il in index_list:
    base = 'F:/Desktop/text/'+ str(il)+ 'q2010/'
    # target = '/Users/cherry/Desktop/zengli/'+ str(il)+ 'q2010plain/'  
    out = 'F:/Desktop/text/new/JigSaw/'+ str(il)+ 'q2010 output of jigsaw/'
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
            # Step 1：尝试使用volume leakage，即寻找每一列中匹配记录频率独特的关键字
            selected_column_sse = ['Hospital','Pincipal Diagnosis']
            matrix_cipher_sse = functions.generate_submatrix(matrix_cipher, selected_column_sse)
            matrix_plain_sse = functions.generate_submatrix(matrix_plain, selected_column_sse)

            # 每个关键字的volume
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

            hospital_plain = functions.extract_columns(matrix_plain, 'Hospital')
            hospital_cipher = functions.extract_columns(matrix_cipher, 'Hospital')
            hospital_list = list(set(hospital_plain).union(set(hospital_cipher)))
            diagnosis_plain = functions.extract_columns(matrix_plain, 'Pincipal Diagnosis')
            diagnosis_cipher = functions.extract_columns(matrix_cipher, 'Pincipal Diagnosis')
            diagnosis_list = list(set(diagnosis_plain).union(set(diagnosis_cipher)))

            list_dh = hospital_list + diagnosis_list

            hospital_frequency_dict = {}

            frequency_folder = 'frequency/'
            for value in hospital_list:
                csv_file_path = os.path.join(frequency_folder, f'{value}.csv')
                if os.path.exists(csv_file_path):
                    value_dict = functions.process_csv_file(csv_file_path)
                    hospital_frequency_dict[value] = value_dict

            # 打印最终的字典
            # print(hospital_frequency_dict)

            # 每个关键字的搜索频率
            hos_freq_cipher_dict = {key: value for key,value in hospital_frequency_dict.items() if key in hospital_cipher}
            hos_freq_plain_dict = {key: value for key,value in hospital_frequency_dict.items() if key in hospital_plain}
            diagnosis_frequency_dict = {}

            for value in diagnosis_list:
                csv_file_path = os.path.join(frequency_folder, f'{value}.csv')
                if os.path.exists(csv_file_path):
                    value_dict = functions.process_csv_file(csv_file_path)
                    diagnosis_frequency_dict[value] = value_dict

            # 每个关键字的搜索频率
            diag_freq_cipher_dict = {key: value for key,value in diagnosis_frequency_dict.items() if key in diagnosis_cipher}
            diag_freq_plain_dict = {key: value for key,value in diagnosis_frequency_dict.items() if key in diagnosis_plain}
            # Step 1 计算v = 匹配上的文档数/总文档数
            hos_volume_cipher_dict = volumn_cipher_sse['Hospital']
            diag_volume_cipher_dict = volumn_cipher_sse['Pincipal Diagnosis']
            hos_volume_plain_dict = volumn_plain_sse['Hospital']
            diag_volume_plain_dict = volumn_plain_sse['Pincipal Diagnosis']

            hos_freq_cipher_dict1, hos_freq_cipher_dict_noT = functions.numToFreqency(hos_freq_cipher_dict)
            hos_freq_plain_dict1, hos_freq_plain_dict_noT = functions.numToFreqency(hos_freq_plain_dict)
            diag_freq_cipher_dict1, diag_freq_cipher_dict_noT = functions.numToFreqency(diag_freq_cipher_dict)
            diag_freq_plain_dict1, diag_freq_plain_dict_noT = functions.numToFreqency(diag_freq_plain_dict)

            # 先利用volume和freqency找到最独特的几个查询，利用不同时期的频率
            # 先利用volume和freqency找到最独特的几个查询，利用的是整个的查询频率
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
            for key1 in set(diagnosis_cipher):
                if key1 in diag_volume_cipher_dict and key1 in diag_freq_cipher_dict_noT:
                    volume1 = diag_volume_cipher_dict[key1]
                    freq1 = diag_freq_cipher_dict_noT[key1]
                    min_score = 100
                    for key2 in set(diagnosis_cipher):
                        if key2 != key1 and key2 in diag_volume_cipher_dict and key2 in diag_freq_cipher_dict_noT:
                            volume2 = diag_volume_cipher_dict[key2]
                            freq2 = diag_freq_cipher_dict_noT[key2]
                            v = abs(volume1 - volume2) *alpha
                            f = abs(freq1 - freq2)*(1-alpha)
                            score = v+f
                            if score < min_score:
                                min_score = score
                    dis_dict_[key1] = min_score
            sorted_dis_dict_ = dict(sorted(dis_dict_.items(), key=lambda item: item[1], reverse=True))
            print(len(set(hospital_cipher)))
            print(len(sorted_dis_dict_))
            # 对上述查询，找关键字
            pred_dict_ = {}
            count = 0
            for key1, value in sorted_dis_dict_.items():
                # count += 1
                # if count == 50:
                #     break
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
                else:
                    volume1 = diag_volume_cipher_dict[key1]
                    freq1 = diag_freq_cipher_dict_noT[key1]
                    min_score = 100
                    candidate = key1
                    for key2 in set(diagnosis_plain):
                        if key2 in diag_volume_plain_dict and key2 in diag_freq_plain_dict_noT:
                            volume2 = diag_volume_plain_dict[key2]
                            freq2 = diag_freq_plain_dict_noT[key2]
                            v = abs(volume1 - volume2) *alpha
                            f = abs(freq1 - freq2)*(1-alpha)
                            score = v+f
                            if score < min_score:
                                min_score = score
                                candidate = key2
                    pred_dict_[key1] = candidate

                # hospital: 
                # 0.3 310个选择50个最独特的，最终出来6个
                # 直接不选，全部都算，最终出来13个
                # 把diagnosis加上 42/949

            count = 0
            for key,value in pred_dict_.items():
                if key == value:
                    value_mapping_od[key] = value
                    count += 1
            print(count)
            print(len(value_mapping_od))

            file_extension = os.path.splitext(file_name)[1]
            new_file_name = file_name.replace(file_extension, ".txt")
            outputPath = os.path.join(out,new_file_name)
            print(outputPath)
            with open(outputPath,"w") as f:
                f.write("keywords number: " + str(keyword_number) + " successfully recovered keyword number: " + str(len(value_mapping_od)))


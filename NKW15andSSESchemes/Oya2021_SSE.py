import csv
from tqdm import tqdm
import random
from collections import Counter
import sys
sys.path.append("/media/ices/machenrry/zl/Attack for DataBlinder") 
import functions
import numpy as np
import os
from collections import defaultdict
import math
from scipy.optimize import linear_sum_assignment as hungarian
import time

def Oya2021Attack(matrix_cipher, matrix_plain):
    startTime = time.time()

    # matrix_cipher = functions.read_csv_to_matrix(filePath)
    print("------sse-----")
    selected_column_sse = ['Hospital','Pincipal Diagnosis']
    matrix_cipher_sse = functions.generate_submatrix(matrix_cipher, selected_column_sse)
    matrix_plain_sse = functions.generate_submatrix(matrix_plain, selected_column_sse)
    # Query Frequency hospital 
    hospital_cipher = functions.extract_columns(matrix_cipher, 'Hospital')
    hospital_cipher_list = list(set(hospital_cipher))
    # cipher_list = list(set(hospital_cipher_list + diagnosis_cipher_list))

    hospital_frequency_cipher_dict = {}

    frequency_folder = '/media/ices/machenrry/zl/Attack for DataBlinder/frequency/'
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

    endTime = time.time()
    totalTime = round(endTime - startTime, 2)

    accuracy = 0
    count = 0
    for key, value in pred_dict_.items():
        if key == value:
            count += 1
    keyword_count = functions.count_keywords(matrix_cipher_sse[1:]) # 一共有多少个关键字
    accuracy = count / keyword_count
    return totalTime, accuracy

if __name__ == '__main__':
    root = "dataset/text_508029.csv"
    # base = [500, 725, 1050, 1525, 2210, 3205, 4645, 6735, 9765, 14160, 20530, 29770, 43170, 62600, 90750, 131600, 190850, 276750, 401300, 508029]
    base = [500, 725, 1050, 1525, 2210, 3205, 4645, 6735, 9765, 14160, 20530, 29770, 43170, 62600, 90750, 508029]
    matrix = functions.read_csv_to_matrix(root)

    filePathPlain = "dataset/2015.csv"

    out = 'result/single/output of SSEAttack-Oya2021.txt'
    matrix_plain = functions.read_csv_to_matrix(filePathPlain)
    # 打开输出文件
    with open(out, 'w', encoding='utf-8') as f:
        for i in base:
            print(i)
            # 每个文件运行10次取平均值
            total_times = []
            accuracies = []
            for _ in range(10):
                matrix_cipher = functions.random_extract(matrix, i)
                totalTime, accuracy = Oya2021Attack(matrix_cipher, matrix_plain)
                total_times.append(totalTime)
                accuracies.append(accuracy)
            
            # 计算平均值
            avg_time = sum(total_times) / 10
            avg_accuracy = sum(accuracies) / 10
            # 将结果写入文件
            f.write(f"文件路径: 4q2010\\text {i}.txt, 执行时间: {avg_time}秒, 准确率: {avg_accuracy}\n")
            f.write("-" * 50 + "\n")
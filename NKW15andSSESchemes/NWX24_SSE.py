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

def get_sub_matrix(V_cipher, cipher_list, knownQ):
    """
    从矩阵V_cipher中按照knownQ筛选列，获取子矩阵
    :param V_cipher: 原始矩阵，以二维列表形式表示
    :param cipher_list: 关键字列表，对应矩阵行列代表的关键字
    :param knownQ: 用于筛选矩阵列的关键字列表
    :return: 筛选后的子矩阵，同样为二维列表形式
    """
    # 获取列索引
    column_indices = [cipher_list.index(keyword) for keyword in knownQ if keyword in cipher_list]
    sub_matrix = []
    for row in V_cipher:
        sub_row = [row[index].item() for index in column_indices]
        sub_matrix.append(sub_row)
    return sub_matrix

def NKW2024Attack(matrix_cipher, matrix_plain):
    startTime = time.time()

    # matrix_cipher = functions.read_csv_to_matrix(filePath)
    # SSE
    # Step 1：尝试使用volume leakage，即寻找每一列中匹配记录频率独特的关键字
    selected_column_sse = ['Hospital','Pincipal Diagnosis']
    matrix_cipher_sse = functions.generate_submatrix(matrix_cipher, selected_column_sse)
    matrix_plain_sse = functions.generate_submatrix(matrix_plain, selected_column_sse)
    value_mapping_od = {}
    # 每个关键字的volume
    volumn_cipher_sse = functions.count_frequency_multicol(np.array(matrix_cipher_sse[1:]), matrix_cipher_sse[0])
    volumn_plain_sse = functions.count_frequency_multicol(np.array(matrix_plain_sse[1:]), matrix_plain_sse[0])
    mapping_sse = {}
    for key, dict1 in volumn_cipher_sse.items():
        dict2 = volumn_plain_sse[key]
        temp = functions.find_closest_mapping(dict1, dict2)
        # value_mapping.update(temp)
        mapping_sse.update(temp)

    for key, value in mapping_sse.items():
        if key == value:
            value_mapping_od[key] = value
            
    selected_columns_noid = ['Age', 'Admission Type', 'Length of stay', 'Risk', 'Gender', 'Race','Hospital', 'Pincipal Diagnosis']
    keyword_number = functions.count_keywords(functions.generate_submatrix(matrix_cipher, selected_columns_noid)[1:])

    hospital_plain = functions.extract_columns(matrix_plain, 'Hospital')
    hospital_cipher = functions.extract_columns(matrix_cipher, 'Hospital')
    hospital_list = list(set(hospital_plain).union(set(hospital_cipher)))
    diagnosis_plain = functions.extract_columns(matrix_plain, 'Pincipal Diagnosis')
    diagnosis_cipher = functions.extract_columns(matrix_cipher, 'Pincipal Diagnosis')
    diagnosis_list = list(set(diagnosis_plain).union(set(diagnosis_cipher)))

    list_dh = hospital_list + diagnosis_list

    hospital_frequency_dict = {}

    frequency_folder = '/media/ices/machenrry/zl/Attack for DataBlinder/frequency/'
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

    endTime = time.time()
    totalTime = round(endTime - startTime, 2)
    value_mapping = {}
    for row in pred_dict_:
        key = row[0]
        value = row[1]
        value_mapping[key] = value

    for key, value in value_mapping.items():
        if key == value:
            count += 1
    keyword_count = functions.count_keywords(matrix_cipher_sse[1:]) # 一共有多少个关键字
    accuracy = count / keyword_count
    return totalTime, accuracy

if __name__ == '__main__':
    root = "dataset/text_508029.csv"
    filePathPlain = "dataset/2015.csv"
    # quarters = ['4q2010']
    out = 'result/single/output of SSEAttack-NWX24.txt'
    matrix_plain = functions.read_csv_to_matrix(filePathPlain)
    # base = [500, 725, 1050, 1525, 2210, 3205, 4645, 6735, 9765, 14160, 20530, 29770, 43170, 62600, 90750, 131600, 190850, 276750, 401300, 508029]
    base = [276750, 401300, 508029]
    matrix = functions.read_csv_to_matrix(root)
    # 打开输出文件
    with open(out, 'w', encoding='utf-8') as f:
        for i in base:
            print(i)

            # 每个文件运行10次取平均值
            total_times = []
            accuracies = []
            for _ in range(50):
                matrix_cipher = functions.random_extract(matrix, i)
                totalTime, accuracy = NKW2024Attack(matrix_cipher, matrix_plain)
                total_times.append(totalTime)
                accuracies.append(accuracy)
            
            # 计算平均值
            avg_time = sum(total_times) / 50
            avg_accuracy = sum(accuracies) / 50
            # 将结果写入文件
            f.write(f"文件路径: 4q2010\\text {i}.txt, 执行时间: {totalTime}秒, 准确率: {accuracy}\n")

            f.write("-" * 50 + "\n")
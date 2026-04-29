import sys
sys.path.append("/CCS2026/") 
import functions
import pandas as pd
import numpy as np
from collections import Counter
import time
import os
import networkx as nx
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

def flatten_and_count(lst):
    flattened = []
    for sub_lst in lst:
        flattened.extend(sub_lst)
    histogram = {}
    for element in flattened:
        histogram[element] = histogram.get(element, 0) + 1
    return histogram



def l2_optimization_attack(plaintext_hist, ciphertext_hist):
    """
    支持明文与密文空间大小不同的ℓ₂-优化攻击（通过填充虚拟元素对齐空间）
    :param plaintext_hist: 明文直方图（字典，key=明文，value=频率）
    :param ciphertext_hist: 密文直方图（字典，key=密文，value=频率）
    :return: 密文到明文的映射字典（过滤虚拟元素）
    """
    # 提取真实明文和密文
    real_plaintexts = list(plaintext_hist.keys())
    real_ciphertexts = list(ciphertext_hist.keys())
    n_plain = len(real_plaintexts)
    n_cipher = len(real_ciphertexts)
    
    # 确定目标维度（取两者中的较大值）
    target_size = max(n_plain, n_cipher)
    
    # 填充虚拟元素使两者维度一致（文档1-117、1-134节方法）
    padded_plaintexts = real_plaintexts.copy()
    padded_ciphertexts = real_ciphertexts.copy()
    
    # 填充明文（若明文空间更小）
    if n_plain < target_size:
        for i in range(target_size - n_plain):
            virtual_p = f"_virtual_p_{i}"  # 虚拟明文标识
            padded_plaintexts.append(virtual_p)
            plaintext_hist[virtual_p] = 0  # 虚拟元素频率为0
    
    # 填充密文（若密文空间更小）
    if n_cipher < target_size:
        for i in range(target_size - n_cipher):
            virtual_c = f"_virtual_c_{i}"  # 虚拟密文标识
            padded_ciphertexts.append(virtual_c)
            ciphertext_hist[virtual_c] = 0  # 虚拟元素频率为0
    
    # 构建代价矩阵（欧氏距离平方）
    cost_matrix = np.zeros((target_size, target_size))
    for i, c in enumerate(padded_ciphertexts):
        for j, p in enumerate(padded_plaintexts):
            freq_diff = ciphertext_hist[c] - plaintext_hist[p]
            cost_matrix[i, j] = freq_diff **2  # 文档1-126节ℓ₂范数转换
    
    # 匈牙利算法求解最优匹配
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # 生成映射并过滤虚拟元素
    mapping = {}
    for c_idx, p_idx in zip(row_indices, col_indices):
        ciphertext = padded_ciphertexts[c_idx]
        plaintext = padded_plaintexts[p_idx]
        # 仅保留真实元素的映射
        if not (ciphertext.startswith("_virtual_c") or plaintext.startswith("_virtual_p")):
            mapping[ciphertext] = plaintext
    
    return mapping
def FrequencyAnalysisAttack(matrix_cipher, matrix_plain, selected_columns_det):
    startTime = time.time()
    
    # 生成密文子矩阵
    # matrix_cipher = functions.read_csv_to_matrix(filePath)
    matrix_cipher_det = functions.generate_submatrix(matrix_cipher, selected_columns_det)
    keyword_count = functions.count_keywords(matrix_cipher_det[1:])
    
    # 生成明文子矩阵
    matrix_plain_det = functions.generate_submatrix(matrix_plain, selected_columns_det)
    functions.replace_nested_inplace(matrix_plain_det, 'American_Indian/Eskimo/Aleut', 'American Indian/Eskimo/Aleut')
    functions.replace_nested_inplace(matrix_plain_det, 'Asian_or_Pacific_Islander', 'Asian or Pacific Islander')

    # 计算直方图
    element_cipher_det = flatten_and_count(matrix_cipher_det[1:])
    element_plain_det = flatten_and_count(matrix_plain_det[1:])

    value_mapping = l2_optimization_attack(element_plain_det, element_cipher_det)

    endTime = time.time()
    totalTime = round(endTime - startTime, 2)

    accuracy = 0
    count = 0
    for key, value in value_mapping.items():
        if key == value:
            count += 1

    accuracy = count / keyword_count
    return value_mapping, totalTime, accuracy, keyword_count, count

    

if __name__ == '__main__':
    root = "dataset/text_508029.csv"
    filePathPlain = "dataset/2015.csv"
    quarters = ['4q2010']
    out = 'result/single/output of FrequencyAnalysisAttack-NKW15+age.txt'
    matrix_plain = functions.read_csv_to_matrix(filePathPlain)
    base = [500, 725, 1050, 1525, 2210, 3205, 4645, 6735, 9765, 14160, 20530, 29770, 43170, 62600, 90750, 131600, 190850, 276750, 401300, 508029]
    matrix = functions.read_csv_to_matrix(root)

    with open(out, 'w', encoding='utf-8') as f:
        for i in base:
            print(i)
            selected_columns_det = ['Gender', 'Race', 'Age']
            total_times = []
            accuracies = []
            counts = []
            for _ in range(50):
                matrix_cipher = functions.random_extract(matrix, i)
                mapping, totalTime, accuracy ,keyword_count, count = FrequencyAnalysisAttack(matrix_cipher, matrix_plain, selected_columns_det)
                total_times.append(totalTime)
                accuracies.append(accuracy)
                counts.append(count)
                            # 计算平均值
            avg_time = sum(total_times) / 50
            avg_accuracy = sum(accuracies) / 50
            avg_count = sum(counts) / 50
            print(avg_accuracy)
            print("---------------------")
            # 将结果写入文件
            f.write(f"文件路径: 4q2010\\text {i}.txt, 执行时间: {avg_time}秒, 准确率: {avg_accuracy}, 关键字数量: {keyword_count}, 实际恢复: {avg_count}\n")
            f.write("-" * 50 + "\n")

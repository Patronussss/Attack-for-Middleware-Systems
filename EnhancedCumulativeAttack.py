import sys
sys.path.append("F:/Desktop/Supplementary experiments") 
import functions
import pandas as pd
import numpy as np
from collections import Counter
import time
import os
import networkx as nx
from tqdm import tqdm

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def custom_sort_key(item):
    if is_number(item):
        return (0, float(item))
    return (1, item)


def compute_statistic(data):
    freq = Counter(data)
    total = sum(freq.values())
    cdf = {}
    cumulative = 0

    for key in sorted(freq):
        count = freq[key]
        cumulative += count
        freq[key] = (count / total)
        cdf[key] = (cumulative / total)

    return freq, cdf

# 计算明文和密文的共现频率，然后算相似度
def compute_cooccurrence_matrix(data):
    """
    计算数据集的共现矩阵
    :param data: 以列表形式存储的矩阵数据集
    :return: 共现矩阵和关键字到索引的映射
    """
    all_keys = set()
    for row in data[1:]:  # 跳过表头
        all_keys.update(row)
    num_keys = len(all_keys)
    key_index = {key: idx for idx, key in enumerate(all_keys)}
    cooc_matrix = np.zeros((num_keys, num_keys))

    for row in data[1:]:  # 跳过表头
        unique_keys = set(row)
        for key1 in unique_keys:
            idx1 = key_index[key1]
            for key2 in unique_keys:
                idx2 = key_index[key2]
                cooc_matrix[idx1][idx2] += 1

    # 归一化共现矩阵
    if len(data) > 1:
        cooc_matrix /= (len(data) - 1)
    return cooc_matrix, key_index

def pad_vector(vec, target_length):
    """
    对向量进行填充，使其长度达到目标长度
    :param vec: 输入向量
    :param target_length: 目标长度
    :return: 填充后的向量
    """
    if len(vec) < target_length:
        padding = [0] * (target_length - len(vec))
        return np.concatenate((vec, padding))
    return vec

def cooccurrence_similarity(vec_A, vec_B):
    """
    计算两个共现向量的相似度，并转换为代价项
    :param vec_A: 密文的共现向量
    :param vec_B: 明文的共现向量
    :return: 相似度代价项
    """
    max_length = max(len(vec_A), len(vec_B))
    vec_A = pad_vector(vec_A, max_length)
    vec_B = pad_vector(vec_B, max_length)
    # 绝对值
    return np.sum(np.abs(vec_A - vec_B))
    # 欧氏距离
    # return np.linalg.norm(vec_A - vec_B)

def EnhancedCumulativeAttack(filePath, matrix_plain, selected_columns_ope_withoutid):
    startTime = time.time()
    matrix_cipher = functions.read_csv_to_matrix(filePath)
    # selected_columns_ope_withoutid = ['Age', 'Admission Type', 'Length of stay', 'Risk']
    matrix_cipher_ope = functions.generate_submatrix(matrix_cipher, selected_columns_ope_withoutid)
    keyword_count = functions.count_keywords(matrix_cipher_ope[1:]) # 一共有多少个关键字
    
    matrix_plain_ope = functions.generate_submatrix(matrix_plain, selected_columns_ope_withoutid)

    keyword_plain_set = set()
    for row in matrix_plain_ope[1:]:
        for element in row:
            keyword_plain_set.add(element)

    keyword_cipher_set = set()
    for row in matrix_cipher_ope[1:]:
        for element in row:
            keyword_cipher_set.add(element)

    cooc_cipher, index_cipher = compute_cooccurrence_matrix(matrix_cipher_ope)
    cooc_plain, index_plain = compute_cooccurrence_matrix(matrix_plain_ope)

    cdf_cipher = {}
    freq_cipher = Counter()

    cdf_plain = {}
    freq_plain = Counter()

    for i in selected_columns_ope_withoutid:
        column_cipher = functions.extract_columns(matrix_cipher_ope, i)
        freq_i_cipher, cdf_i_cipher = compute_statistic(column_cipher)
        freq_cipher.update(freq_i_cipher)
        cdf_cipher.update(cdf_i_cipher)

        column_plain = functions.extract_columns(matrix_plain_ope, i)
        freq_i_plain, cdf_i_plain = compute_statistic(column_plain)
        freq_plain.update(freq_i_plain)
        cdf_plain.update(cdf_i_plain)

    cost_matrix = np.full((len(keyword_cipher_set), len(keyword_plain_set)), np.inf)
    headers_cipher = matrix_cipher_ope[0]
    headers_plain = matrix_plain_ope[0]
    column_num_cipher = len(matrix_cipher_ope[0])
    column_num_plain = len(matrix_plain_ope[0])

    # 计算代价矩阵部分添加进度条
    for col_cipher in tqdm(range(column_num_cipher), desc="Calculating cost matrix"):
        for col_plain in range(column_num_plain):
            if headers_cipher[col_cipher] == headers_plain[col_plain]:
                    # 获取该列的所有关键字
                    keys_A_in_col = set([row[col_cipher] for row in matrix_cipher_ope[1:]])
                    keys_B_in_col = set([row[col_plain] for row in matrix_plain_ope[1:]])
                    for key_A in keys_A_in_col:
                        idx_A = index_cipher[key_A]
                        for key_B in keys_B_in_col:
                            idx_B = index_plain[key_B]
                            vec_A = cooc_cipher[idx_A]
                            vec_B = cooc_plain[idx_B]
                            similarity = cooccurrence_similarity(vec_A, vec_B)
                            freq_diff = abs(freq_cipher[key_A] - freq_plain[key_B])
                            cdf_diff = abs(cdf_cipher[key_A] - cdf_plain[key_B])
                            cost_matrix[idx_A][idx_B] = 1 * freq_diff + 1 * cdf_diff  - 100 * similarity

    # 创建一个有向图
    G = nx.DiGraph()
    source = 'source'
    sink = 'sink'
    # 添加源节点到密文关键字节点的边
    for i, key_A in enumerate(keyword_cipher_set):
        G.add_edge(source, key_A, capacity=1, weight=0)
    # 添加密文关键字节点到明文关键字节点的边
    for i, key_A in tqdm(enumerate(keyword_cipher_set), desc="Adding edges to graph"):
        for j, key_B in enumerate(keyword_plain_set):
            if cost_matrix[i][j] != np.inf:
                G.add_edge(key_A, key_B, capacity=1, weight=cost_matrix[i][j])
    # 添加明文关键字节点到汇节点的边
    for j, key_B in enumerate(keyword_plain_set):
        G.add_edge(key_B, sink, capacity=1, weight=0)
    # 计算最小费用流
    flow_dict = nx.min_cost_flow(G, demand=-len(keyword_cipher_set))
    # 构建映射关系
    value_mapping = {}
    for key_A in keyword_cipher_set:
        for key_B in flow_dict[key_A]:
            if flow_dict[key_A][key_B] > 0:
                value_mapping[key_A] = key_B

    endTime = time.time()
    totalTime = round(endTime - startTime, 2)

    accuracy = 0
    count = 0
    for key, value in value_mapping.items():
        if key == value:
            count += 1

    accuracy = count / keyword_count
    return value_mapping, totalTime, accuracy, keyword_count

if __name__ == '__main__':
    # root = "F:/Desktop/Attack for Datablinder/4q2010/text_508029.csv"
    # filePathPlain = "F:/Desktop/Attack for Datablinder/2015/2015.csv"
    # matrix_plain = functions.read_csv_to_matrix(filePathPlain)
    # mapping, totalTime, accuracy ,keyword_count, count = EnhancedCumulativeAttack(root, matrix_plain)
    # print(accuracy)
    root = "F:/Desktop/Supplementary experiments/New Dataset/"
    filePathPlain = "F:/Desktop/Attack for Datablinder/2015/2015.csv"
    quarters = ['4q2010']
    # quarters = ['1q2010', '2q2010', '3q2010', '4q2010']
    out = 'result/20250421/output of CumulativeAttack-ours.txt'
    matrix_plain = functions.read_csv_to_matrix(filePathPlain)
    # 打开输出文件
    with open(out, 'w', encoding='utf-8') as f:
        for base in quarters:
            rootPath = os.path.join(root,base)
            file_list = os.listdir(rootPath)
            for file_name in file_list:
                filePath = os.path.join(rootPath,file_name)
                # filePathPlain = os.path.join(rootPath,file_name)
                name = os.path.join(base, file_name)
                print(name)
                selected_columns_ope_withoutid = ['Age', 'Admission Type', 'Length of stay', 'Risk']
                # 每个文件运行10次取平均值
                total_times = []
                accuracies = []
                for _ in range(10):
                    mapping, totalTime, accuracy ,keyword_count = EnhancedCumulativeAttack(filePath, matrix_plain, selected_columns_ope_withoutid)
                    total_times.append(totalTime)
                    accuracies.append(accuracy)
                
                # 计算平均值
                avg_time = sum(total_times) / 10
                avg_accuracy = sum(accuracies) / 10
                # 将结果写入文件
                f.write(f"文件路径: {name}, 执行时间: {totalTime}秒, 准确率: {accuracy}, 关键字数量: {keyword_count}, 实际恢复: {len(mapping) * accuracy}\n")
                f.write("-" * 50 + "\n")
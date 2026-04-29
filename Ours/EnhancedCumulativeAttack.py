import sys
sys.path.append("/media/ices/machenrry/zl/Attack for DataBlinder") 
import functions
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
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

# 新：计算每个元素与其他元素的共现频率（不生成矩阵，直接用字典存储）
def compute_cooccurrence_freq(data):
    """
    计算每个元素与其他元素在同一记录（行）中的共现频率
    :param data: 矩阵数据集（含表头）
    :return: 共现频率字典 {元素a: {元素b: 共现频率, ...}, ...}
    """
    # 存储每个元素与其他元素的共现次数
    cooc_counts = defaultdict(Counter)
    # 总记录数（行数，跳过表头）
    total_records = len(data) - 1 if len(data) > 1 else 0
    
    if total_records == 0:
        return dict(cooc_counts)  # 空字典
    
    # 遍历每一行（记录）
    for row in data[1:]:  # 跳过表头
        # 去重当前行的元素（避免同一行中同一元素重复统计）
        unique_elements = list(set(row))
        # 统计当前行中所有元素对的共现
        for i in range(len(unique_elements)):
            elem_a = unique_elements[i]
            for j in range(len(unique_elements)):
                if i != j:  # 只统计不同元素的共现
                    elem_b = unique_elements[j]
                    cooc_counts[elem_a][elem_b] += 1  # 共现次数+1
    
    # 将次数转换为频率（除以总记录数）
    cooc_freq = {}
    for elem_a, counts in cooc_counts.items():
        freq_dict = {elem_b: count / total_records for elem_b, count in counts.items()}
        cooc_freq[elem_a] = freq_dict
    
    return cooc_freq

def cooccurrence_distance(elem_a, elem_b, cooc_freq_a_dict, cooc_freq_b_dict):
    """
    计算两个元素的共现距离（基于它们与其他元素的共现频率）
    :param elem_a: 密文元素
    :param elem_b: 明文元素
    :param cooc_freq_a_dict: 密文共现频率字典 {元素: {其他元素: 频率}}
    :param cooc_freq_b_dict: 明文共现频率字典 {元素: {其他元素: 频率}}
    :return: 共现距离（值越大，共现模式差异越大）
    """
    # 获取elem_a的共现频率（默认0）
    a_cooc = cooc_freq_a_dict.get(elem_a, {})
    # 获取elem_b的共现频率（默认0）
    b_cooc = cooc_freq_b_dict.get(elem_b, {})
    
    # 收集所有需要比较的元素（a的共现元素 + b的共现元素）
    all_related_elements = set(a_cooc.keys()).union(set(b_cooc.keys()))
    
    # 计算频率差异的总和（曼哈顿距离）
    distance = 0.0
    for elem in all_related_elements:
        freq_a = a_cooc.get(elem, 0.0)  # a与elem的共现频率
        freq_b = b_cooc.get(elem, 0.0)  # b与elem的共现频率
        distance += abs(freq_a - freq_b)
    
    return distance

def EnhancedCumulativeAttack(matrix_cipher, matrix_plain, selected_columns_ope_withoutid, weight):
    startTime = time.time()
    a = weight[0]
    b = weight[1]
    c = weight[2]
    # 提取密文和明文的目标列数据
    matrix_cipher_ope = functions.generate_submatrix(matrix_cipher, selected_columns_ope_withoutid)
    keyword_count = functions.count_keywords(matrix_cipher_ope[1:])  # 关键字总数
    
    matrix_plain_ope = functions.generate_submatrix(matrix_plain, selected_columns_ope_withoutid)

    # 提取密文和明文的所有关键字
    keyword_plain_set = set()
    for row in matrix_plain_ope[1:]:
        for element in row:
            keyword_plain_set.add(element)

    keyword_cipher_set = set()
    for row in matrix_cipher_ope[1:]:
        for element in row:
            keyword_cipher_set.add(element)

    # 计算共现频率（新方法：不生成矩阵，直接用字典）
    cooc_freq_cipher = compute_cooccurrence_freq(matrix_cipher_ope)  # 密文共现频率
    cooc_freq_plain = compute_cooccurrence_freq(matrix_plain_ope)    # 明文共现频率

    # 计算频率和CDF
    cdf_cipher = {}
    freq_cipher = Counter()
    cdf_plain = {}
    freq_plain = Counter()

    for col in selected_columns_ope_withoutid:
        column_cipher = functions.extract_columns(matrix_cipher_ope, col)
        freq_i_cipher, cdf_i_cipher = compute_statistic(column_cipher)
        freq_cipher.update(freq_i_cipher)
        cdf_cipher.update(cdf_i_cipher)

        column_plain = functions.extract_columns(matrix_plain_ope, col)
        freq_i_plain, cdf_i_plain = compute_statistic(column_plain)
        freq_plain.update(freq_i_plain)
        cdf_plain.update(cdf_i_plain)

    # 初始化代价矩阵
    cost_matrix = np.full((len(keyword_cipher_set), len(keyword_plain_set)), np.inf)
    headers_cipher = matrix_cipher_ope[0]
    headers_plain = matrix_plain_ope[0]
    column_num_cipher = len(matrix_cipher_ope[0])
    column_num_plain = len(matrix_plain_ope[0])

    # 构建密文/明文关键字到索引的映射（用于代价矩阵索引）
    cipher_to_idx = {key: idx for idx, key in enumerate(keyword_cipher_set)}
    plain_to_idx = {key: idx for idx, key in enumerate(keyword_plain_set)}

    # 计算代价矩阵（使用新的共现距离）
    for col_cipher in tqdm(range(column_num_cipher), desc="Calculating cost matrix"):
        for col_plain in range(column_num_plain):
            if headers_cipher[col_cipher] == headers_plain[col_plain]:
                # 获取当前列的密文和明文关键字
                keys_A_in_col = set([row[col_cipher] for row in matrix_cipher_ope[1:]])
                keys_B_in_col = set([row[col_plain] for row in matrix_plain_ope[1:]])
                
                for key_A in keys_A_in_col:
                    idx_A = cipher_to_idx[key_A]
                    for key_B in keys_B_in_col:
                        idx_B = plain_to_idx[key_B]
                        # 计算共现距离（基于与其他元素的共现频率）
                        cooc_dist = cooccurrence_distance(
                            key_A, key_B, 
                            cooc_freq_cipher, cooc_freq_plain
                        )
                        # 频率差异和CDF差异
                        freq_diff = abs(freq_cipher.get(key_A, 0) - freq_plain.get(key_B, 0))
                        cdf_diff = abs(cdf_cipher.get(key_A, 0) - cdf_plain.get(key_B, 0))
                        # 总代价（差异越大，代价越高）
                        cost_matrix[idx_A][idx_B] = a * freq_diff  + b * cdf_diff  + c * cooc_dist
                        # a=0.5, b=0.1, c=0.4

    # 创建有向图用于最小费用流
    G = nx.DiGraph()
    source = 'source'
    sink = 'sink'
    # 添加源节点到密文关键字节点的边
    for key_A in keyword_cipher_set:
        G.add_edge(source, key_A, capacity=1, weight=0)
    # 添加密文关键字节点到明文关键字节点的边
    for i, key_A in tqdm(enumerate(keyword_cipher_set), desc="Adding edges to graph"):
        for j, key_B in enumerate(keyword_plain_set):
            if cost_matrix[i][j] != np.inf:
                G.add_edge(key_A, key_B, capacity=1, weight=cost_matrix[i][j])
    # 添加明文关键字节点到汇节点的边
    for key_B in keyword_plain_set:
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

    # 计算准确率
    correct_count = 0
    for key, value in value_mapping.items():
        if key == value:
            correct_count += 1
    accuracy = correct_count / keyword_count if keyword_count > 0 else 0

    return value_mapping, totalTime, accuracy, keyword_count

if __name__ == '__main__':
    root = "dataset/text_508029.csv"
    filePathPlain = "dataset/2015.csv"
    out = 'result/single/output of CumulativeAttack-ours_110.txt'
    matrix_plain = functions.read_csv_to_matrix(filePathPlain)

    # base = [508029]
    base = [500, 725, 1050, 1525, 2210, 3205, 4645, 6735, 9765, 14160, 20530, 29770, 43170, 62600, 90750, 131600, 190850, 276750, 401300, 508029]
    matrix = functions.read_csv_to_matrix(root)

    with open(out, 'w', encoding='utf-8') as f:
        for i in base:
            print(i)
            selected_columns_ope_withoutid = ['Age', 'Admission Type', 'Length of stay', 'Risk']
            total_times = []
            accuracies = []
            keyword_counts = []
            # 每个文件运行1次
            for _ in range(50):
                matrix_cipher = functions.random_extract(matrix, i)
                mapping, totalTime, accuracy, keyword_count = EnhancedCumulativeAttack(matrix_cipher, matrix_plain, selected_columns_ope_withoutid, [0.5, 0.1, 0.4])
                total_times.append(totalTime)
                accuracies.append(accuracy)
                keyword_counts.append(keyword_count)

            # 计算平均值
            avg_time = sum(total_times) / 50
            avg_accuracy = sum(accuracies) / 50
            avg_count = sum(keyword_counts) / 50
            print(avg_accuracy)
            # 写入结果
            f.write(f"文件路径: 4q2010\\text {i}.txt, 执行时间: {avg_time}秒, 准确率: {avg_accuracy}, 关键字数量: {avg_count}, 实际恢复: {len(mapping) * avg_accuracy}\n")
            f.write("-" * 50 + "\n")
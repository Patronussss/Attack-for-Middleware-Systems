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

def compute_feature_vector(matrix, select_column, dataset_name):
    # 获取表头
    headers = matrix[0]
    # 找到A列的索引
    a_index = headers.index(select_column)
    if dataset_name == "PUDF":
        specified_values = {
    "Age": ["0 year", "1 year", "5 year", "10 year", "15 year", "18 year",
        "20 year", "25 year", "30 year", "35 year", "40 year", "45 year", "50 year", 
        "55 year", "60 year", "65 year", "70 year", "75 year", "80 year", "85 year", "90 year"],
    # "Gender": ["F", "M"],
    "Risk": ["Minor", "Moderate", "Major", "Extreme"],
    "Admission Type": ["Emergency", "Urgent", "Elective", "Newborn", "Trauma_Center", "Others"]
    # "Race": ["American_Indian/Eskimo/Aleut","Asian_or_Pacific_Islander", 
            # "Black", "White", "Others", "Invalid", "Hispanic"]
    }
        # 定义要统计取值范围的列
        specific_columns = [col for col in ["Age", "Risk", "Admission Type"] if col in headers]
        if not specific_columns:
            specific_columns = [select_column]  # 如果没有其他列，就用当前列本身
    # 获取这些列的索引
    if dataset_name == "Alzheimer":
        specified_values = {
            "YearStart": ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"],
            "LocationAbbr": ["FL", "AR", "LA", "GA", "MA", "NE", "MDW", "MO", "DE", "DC", "IL", "AK", "OK", "SOU",
            "TX", "IA", "MD", "PA", "WI", "AZ", "KS", "ND", "OH", "NM", "UT", "WEST", "NH", "US", "NV", "WA", "RI", "ME", "VT", "ID",
            "AL", "MI", "SD", "CT", "NC", "TN", "WV", "WY", "VA", "NJ", "CA", "OR", "MS", "HI", "NY", "MN", "CO", "PR", "KY", "IN", "MT",
            "NRE", "SC", "GU", "VI"],
            "Stratification2": ["Native am/Alaskan Native", "Asian/Pacific Islander", "Black, non-Hispanic", "White, non-Hispanic", "Hispanic", "Female", "Male"],
            "Class": ["Mental Health", "Overall Health", "Caregiving", "Nutrition/Physical Activity/Obesity", "Screenings and  Vaccines", "Smoking and Alcohol  Use", "Cognitive Decline"],
            "DataValueTypeID" : ["PRCTG", "MEAN"]
        }
        specific_columns = [col for col in ["YearStart", "LocationAbbr", "Stratification2", "Class", "DataValueTypeID"] if col in headers]
        if not specific_columns:
            specific_columns = [select_column]
    if dataset_name == "Crime":
        specified_values = {
            'AREA NAME': ["Wilshire", "Central", "Southwest", "Van Nuys", "Hollywood", "Southeast", "Newton"," Mission", "Rampart", "West Valley", 
            "77th Street", "Devonshire", "Foothill", "Harbor", "Hollenbeck", "N Hollywood", "Northeast", "Olympic", "Pacific", "Topanga", "West LA"],
            'Vict Sex': ['F', 'M', 'H', 'X', '-'],
            'Part 1-2': ['1','2'],
            'Status': ['AA','AO', 'CC', 'IC', 'JA', 'JO']
        }
        specific_columns = [col for col in ["AREA NAME", "Vict Sex", "Part 1-2", "Status"] if col in headers]
        if not specific_columns:
            specific_columns = [select_column]
    if dataset_name == "ACS":
        specified_values = {
        "AGE": ["0 year", "1 year", "5 year", "10 year", "15 year", "18 year",
                "20 year", "25 year", "30 year", "35 year", "40 year", "45 year", "50 year", 
                "55 year", "60 year", "65 year", "70 year", "75 year", "80 year", "85 year", "90 year"],
        "SEX":[1,2],
        "MARST":[1,2,3,4,5,6],
        "RACE":[1,2,3,4,5,6,7,8,9],
        "EDUC":[0,1,2,3,4,5,6,7,8,10,11]
        # "HHINCOME":[]
        }
        specific_columns = [col for col in ['AGE','SEX','MARST','RACE','EDUC'] if col in headers]
        if not specific_columns:
            specific_columns = [select_column]
    specific_indices = [headers.index(col) for col in specific_columns]

    # 初始化一个字典来存储A列每个值对应的特定列的数据
    data_dict = {}
    for row in matrix[1:]:
        a_value = row[a_index]
        if a_value == "zlzlzl":
            continue
        if a_value not in data_dict:
            data_dict[a_value] = {col: [] for col in specific_columns}
        for col, index in zip(specific_columns, specific_indices):
            value = row[index]
            # 若特定列的值为zlzlzl则跳过该值
            if value != "zlzlzl":
                data_dict[a_value][col].append(value)

    # 统计每个A列值对应的特定列的取值比例
    result = {}
    for a_value, col_data in data_dict.items():
        result[a_value] = []
        for col in specific_columns:
            values = col_data[col]
            value_count = len(values)
            value_proportion = {val: values.count(val) / value_count for val in set(values)}
            vector = [value_proportion.get(val, 0) for val in specified_values[col]]
            result[a_value].extend(vector)

    return result

def build_cost_matrix(cipher_matrix, plain_matrix, cipher_features, plain_features, freq_weight=1, feat_weight=1):
    """
    构建代价矩阵（结合频率和特征向量）
    :param cipher_matrix: 密文频率的嵌套字典
    :param plain_matrix: 明文频率的嵌套字典
    :param cipher_features: 密文特征向量的嵌套字典
    :param plain_features: 明文特征向量的嵌套字典
    :param freq_weight: 频率差异的权重
    :param feat_weight: 特征向量差异的权重
    :return: 代价矩阵和关键字到索引的映射
    """
    all_cipher_keys = set()
    all_plain_keys = set()

    # 收集所有密文和明文关键字
    for id_, cipher_freq in cipher_matrix.items():
        all_cipher_keys.update(cipher_freq.keys())
    for id_, plain_freq in plain_matrix.items():
        all_plain_keys.update(plain_freq.keys())

    num_cipher = len(all_cipher_keys)
    num_plain = len(all_plain_keys)
    key_index_cipher = {key: idx for idx, key in enumerate(all_cipher_keys)}
    key_index_plain = {key: idx for idx, key in enumerate(all_plain_keys)}
    cost_matrix = np.full((num_cipher, num_plain), np.inf)

    # 遍历所有列（如Gender、Race、Age）
    for col in cipher_matrix.keys():
        cipher_freq = cipher_matrix[col]
        plain_freq = plain_matrix[col]
        cipher_feat = cipher_features.get(col, {})  # 该列的密文特征向量
        plain_feat = plain_features.get(col, {})    # 该列的明文特征向量

        for key_A, freq_A in cipher_freq.items():
            idx_A = key_index_cipher[key_A]
            # 获取密文关键字的特征向量
            feat_A = np.array(cipher_feat.get(key_A, []), dtype=np.float32)
            
            for key_B, freq_B in plain_freq.items():
                idx_B = key_index_plain[key_B]
                # 获取明文关键字的特征向量
                feat_B = np.array(plain_feat.get(key_B, []), dtype=np.float32)

                # 计算频率差异（绝对值）
                freq_diff = np.abs(freq_A - freq_B)

                # 计算特征向量差异（欧氏距离）
                if len(feat_A) == 0 or len(feat_B) == 0:
                    feat_diff = 0.0  # 无特征向量时差异为0
                else:
                    # 确保向量长度一致
                    min_len = min(len(feat_A), len(feat_B))
                    feat_diff = np.linalg.norm(feat_A[:min_len] - feat_B[:min_len])

                # 综合代价（加权求和后取负值，因为最小费用流找最小值）
                total_diff = (freq_weight * freq_diff) + (feat_weight * feat_diff)
                cost_matrix[idx_A][idx_B] = -total_diff  # 取负值使相似的对代价更小

    return cost_matrix, key_index_cipher, key_index_plain

def find_mapping(cipher_matrix, plain_matrix, cipher_features, plain_features, weight):
    """
    使用最小费用流问题找到明文到密文的映射
    :param cipher_matrix: 密文频率的嵌套字典
    :param plain_matrix: 明文频率的嵌套字典
    :return: 明文到密文的映射
    """
    freq_weight = weight[0]
    feat_weight = weight[1]
    cost_matrix, key_index_cipher, key_index_plain = build_cost_matrix(
        cipher_matrix, plain_matrix, cipher_features, plain_features, freq_weight, feat_weight
    )
    all_cipher_keys = list(key_index_cipher.keys())
    all_plain_keys = list(key_index_plain.keys())
    print(f"    [DET] Building graph: {len(all_cipher_keys)} cipher keys, {len(all_plain_keys)} plain keys")

    # 创建一个有向图
    G = nx.DiGraph()
    source = 'source'
    sink = 'sink'
    # 添加源节点到密文关键字节点的边
    for key_A in all_cipher_keys:
        G.add_edge(source, key_A, capacity=1, weight=0)
    # 添加密文关键字节点到明文关键字节点的边
    for i, key_A in enumerate(all_cipher_keys):
        for j, key_B in enumerate(all_plain_keys):
            if cost_matrix[i][j] != np.inf:
                G.add_edge(key_A, key_B, capacity=1, weight=cost_matrix[i][j])
    # 添加明文关键字节点到汇节点的边
    for key_B in all_plain_keys:
        G.add_edge(key_B, sink, capacity=1, weight=0)
    # print(f"    [DET] Graph built with {len(G.nodes())} nodes, {len(G.edges())} edges")
    # print(f"    [DET] Running min_cost_flow...")
    # 计算最小费用流
    flow_dict = nx.min_cost_flow(G, demand=-len(all_cipher_keys))
    # print(f"    [DET] min_cost_flow completed")
    # 构建映射关系
    value_mapping = {}
    for key_A in all_cipher_keys:
        for key_B in flow_dict[key_A]:
            if flow_dict[key_A][key_B] > 0:
                value_mapping[key_A] = key_B

    return value_mapping

def EnhancedFrequencyAnalysisAttack(matrix_cipher, matrix_plain, selected_columns_det, dataset_name, weight):
    startTime = time.time()
    # print(f"    [DET] Starting attack with columns: {selected_columns_det}")
    
    # 生成密文子矩阵
    # matrix_cipher = functions.read_csv_to_matrix(filePath)
    matrix_cipher_det = functions.generate_submatrix(matrix_cipher, selected_columns_det)
    keyword_count = functions.count_keywords(matrix_cipher_det[1:])
    # print(f"    [DET] Cipher matrix shape: {len(matrix_cipher_det)} rows, keyword count: {keyword_count}")
    
    # 生成明文子矩阵
    matrix_plain_det = functions.generate_submatrix(matrix_plain, selected_columns_det)
    # functions.replace_nested_inplace(matrix_plain_det, 'American_Indian/Eskimo/Aleut', 'American Indian/Eskimo/Aleut')
    # functions.replace_nested_inplace(matrix_plain_det, 'Asian_or_Pacific_Islander', 'Asian or Pacific Islander')
    # print(f"    [DET] Plain matrix shape: {len(matrix_plain_det)} rows")
    
    # 计算频率
    element_cipher_det = functions.column_frequencies(matrix_cipher_det[1:])
    element_plain_det = functions.column_frequencies(matrix_plain_det[1:])
    # print(f"    [DET] Cipher columns: {list(element_cipher_det.keys())}, Plain columns: {list(element_plain_det.keys())}")
    
    # 计算特征向量
    feature_cipher_det = {}
    feature_plain_det = {}
    # 创建列名到列索引的映射
    col_name_to_index = {col: idx for idx, col in enumerate(matrix_cipher_det[0])}
    # print("Feature Vector")
    for col in selected_columns_det:
        if col in col_name_to_index:
            col_idx = col_name_to_index[col]
            feature_cipher_det[col_idx] = compute_feature_vector(matrix_cipher, col, dataset_name)
            feature_plain_det[col_idx] = compute_feature_vector(matrix_plain, col, dataset_name)
        else:
            print(f"    [DET] Column {col} not found in cipher matrix")
    # print(f"    [DET] Feature vectors computed for {len(feature_cipher_det)} columns")
    
    # print(f"    [DET] Running find_mapping...")
    value_mapping = find_mapping(element_cipher_det, element_plain_det, feature_cipher_det, feature_plain_det, weight)
    # print(f"    [DET] Mapping found: {len(value_mapping)} pairs")


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
    
    filePathPlain = "dataset/2015.csv"
    quarters = ['4q2010']
    out = 'result/CCS/output of FrequencyAnalysisAttack-weight.txt'
    matrix_plain = functions.read_csv_to_matrix(filePathPlain)

    root = "dataset/text_508029.csv"
    base = 500000
    matrix = functions.read_csv_to_matrix(root)

    alpha = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
             0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    weights = []
    for a in alpha:
        b = round(1 - a, 2)
        w = [a, b]
        weights.append(w)

    with open(out, 'w', encoding='utf-8') as f:
        for weight in weights:
            print(weight)
            selected_columns_det = ['Gender', 'Race', 'Age', 'Length of stay']
            total_times = []
            accuracies = []
            counts = []
            for _ in range(50):
                matrix_cipher = functions.random_extract(matrix, base)
                mapping, totalTime, accuracy, keyword_count, count = EnhancedFrequencyAnalysisAttack(matrix_cipher, matrix_plain, selected_columns_det, "PUDF", weight)
                total_times.append(totalTime)
                accuracies.append(accuracy)
                counts.append(count)
            avg_time = sum(total_times) / 50
            avg_accuracy = sum(accuracies) / 50
            avg_count = sum(counts) / 50
            print(avg_accuracy)
            print("---------------------")
            f.write(f"权重： {weight}, 执行时间: {avg_time}秒, 准确率: {avg_accuracy}, 关键字数量: {keyword_count}, 实际恢复: {avg_count}\n")
            f.write("-" * 50 + "\n")
    
    

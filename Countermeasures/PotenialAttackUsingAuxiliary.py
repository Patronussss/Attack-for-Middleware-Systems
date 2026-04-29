import sys
sys.path.append("/media/ices/machenrry/zl/Attack for DataBlinder") 
import functions
import pandas as pd
import numpy as np
from collections import Counter
import time
import os
import networkx as nx
from tqdm import tqdm

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
    # 计算最小费用流
    flow_dict = nx.min_cost_flow(G, demand=-len(all_cipher_keys))
    # 构建映射关系
    value_mapping = {}
    for key_A in all_cipher_keys:
        for key_B in flow_dict[key_A]:
            if flow_dict[key_A][key_B] > 0:
                value_mapping[key_A] = key_B

    return value_mapping

def EnhancedFrequencyAnalysisAttack(matrix_cipher, matrix_plain, selected_columns_det, dataset_name):
    startTime = time.time()
    
    # 生成密文子矩阵
    # matrix_cipher = functions.read_csv_to_matrix(filePath)
    matrix_cipher_det = functions.generate_submatrix(matrix_cipher, selected_columns_det)
    keyword_count = functions.count_keywords(matrix_cipher_det[1:])
    
    # 生成明文子矩阵
    matrix_plain_det = functions.generate_submatrix(matrix_plain, selected_columns_det)
    # functions.replace_nested_inplace(matrix_plain_det, 'American_Indian/Eskimo/Aleut', 'American Indian/Eskimo/Aleut')
    # functions.replace_nested_inplace(matrix_plain_det, 'Asian_or_Pacific_Islander', 'Asian or Pacific Islander')
    
    # 计算频率
    element_cipher_det = functions.column_frequencies(matrix_cipher_det[1:])
    element_plain_det = functions.column_frequencies(matrix_plain_det[1:])

    # 计算特征向量
    feature_cipher_det = {}
    feature_plain_det = {}
    for col in selected_columns_det:
        feature_cipher_det[col] = compute_feature_vector(matrix_cipher, col, dataset_name)
        feature_plain_det[col] = compute_feature_vector(matrix_plain, col, dataset_name)
    
    
    # value_mapping = {}
    # for key, dict1 in element_cipher_det.items():
    #     dict2 = element_plain_det[key]
    #     temp = functions.find_closest_mapping(dict1, dict2)
    #     value_mapping.update(temp)

    value_mapping = find_mapping(element_cipher_det, element_plain_det, feature_cipher_det, feature_plain_det, [1,0])



    endTime = time.time()
    totalTime = round(endTime - startTime, 2)

    accuracy = 0
    count = 0
    for key, value in value_mapping.items():
        if key == value:
            count += 1

    accuracy = count / keyword_count
    return value_mapping, totalTime, accuracy, keyword_count, count



import random  # 导入random模块用于生成随机数

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
    "Gender": ["F", "M"],
    "Risk": ["Minor", "Moderate", "Major", "Extreme"],
    "Admission Type": ["Emergency", "Urgent", "Elective", "Newborn", "Trauma_Center", "Others"],
    "Race": ["American_Indian/Eskimo/Aleut","Asian_or_Pacific_Islander", 
            "Black", "White", "Others", "Invalid", "Hispanic"]
    }
        # 定义要统计取值范围的列
        specific_columns = ["Age", "Gender",  "Risk", "Admission Type", "Race"]
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
        specific_columns = ["YearStart", "LocationAbbr",  "Stratification2", "Class", "DataValueTypeID"]
    if dataset_name == "Crime":
        specified_values = {
            'AREA NAME': ["Wilshire", "Central", "Southwest", "Van Nuys", "Hollywood", "Southeast", "Newton"," Mission", "Rampart", "West Valley", 
            "77th Street", "Devonshire", "Foothill", "Harbor", "Hollenbeck", "N Hollywood", "Northeast", "Olympic", "Pacific", "Topanga", "West LA"],
            'Vict Sex': ['F', 'M', 'H', 'X', '-'],
            'Part 1-2': ['1','2'],
            'Status': ['AA','AO', 'CC', 'IC', 'JA', 'JO']
        }
        specific_columns = ["AREA NAME", "Vict Sex",  "Part 1-2", "Status"]
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

    # 统计每个A列值对应的特定列的取值比例（修改为随机数）
    result = {}
    for a_value, col_data in data_dict.items():
        result[a_value] = []
        for col in specific_columns:
            # 生成与specified_values[col]元素数量相同的随机数向量
            # 这里使用0到1之间的随机浮点数，你可以根据需要调整范围
            vector = [random.random() for _ in specified_values[col]]
            # 如果你需要整数，可以使用:
            # vector = [random.randint(0, 100) for _ in specified_values[col]]
            result[a_value].extend(vector)

    return result

def compute_cost_matrix(feature_cipher_dict, freq_cipher_dict, vol_cipher_dict, feature_plain_dict, freq_plain_dict, vol_plain_dict):
    cost_matrix = np.full((len(freq_cipher_dict), len(freq_plain_dict)), np.inf)
    sorted_cipher_keys = sorted(freq_cipher_dict.keys())
    sorted_plain_keys = sorted(freq_plain_dict.keys())

    # 创建密文key到index的映射
    cipher_key_to_idx = {k: i for i, k in enumerate(sorted_cipher_keys)}
    plain_key_to_idx = {k: i for i, k in enumerate(sorted_plain_keys)}

    for keyword in tqdm(sorted_cipher_keys):
        cipher_idx = cipher_key_to_idx[keyword]
        feature_cipher = feature_cipher_dict[keyword]
        freq_cipher = freq_cipher_dict[keyword]
        vol_cipher = vol_cipher_dict[keyword]
        
        for keyword_plain in sorted_plain_keys:
            plain_idx = plain_key_to_idx[keyword_plain]
            feature_plain = feature_plain_dict[keyword_plain]
            freq_plain = freq_plain_dict[keyword_plain]
            vol_plain = vol_plain_dict[keyword_plain]
            
            # 计算三个代价并相加
            # 计算特征向量的欧氏距离
            fea_cost = np.linalg.norm(np.array(feature_cipher) - np.array(feature_plain))
            # 计算频率向量的欧氏距离
            freq_cost = np.linalg.norm(np.array(freq_cipher) - np.array(freq_plain))
            # 计算体积（标量）的绝对差
            vol_cost = (vol_cipher - vol_plain) ** 2
            total_cost = 0 * fea_cost + 1 * freq_cost + 1 * vol_cost
            
            # 按照key的index存储总代价
            cost_matrix[cipher_idx, plain_idx] = total_cost
    return cost_matrix

def auction_algorithm(cost_matrix, epsilon=0.1, max_time=3000, recovery_threshold=0.9):
    """
    拍卖算法实现

    :param cost_matrix: 代价矩阵，shape 为 (n_cipher, n_plain)
    :param epsilon: 投标增量，用于控制投标的步长
    :param max_time: 最大运行时间（秒），超过此时间将提前终止
    :param recovery_threshold: 成功恢复阈值，当恢复率达到此值时提前终止
    :return: 匹配结果，数组中每个元素表示 sorted_cipher_keys 中对应关键字匹配到的 sorted_plain_keys 中的索引
    """
    import time
    start_time = time.time()
    n_cipher, n_plain = cost_matrix.shape
    prices = np.zeros(n_plain)  # 初始化卖家的价格
    assignments = np.full(n_cipher, -1)  # 初始化匹配结果，-1 表示未匹配

    # 计算总迭代次数用于进度条显示
    total_unassigned = np.sum(assignments == -1)
    pbar = tqdm(total=total_unassigned, desc='拍卖算法进行中')
    prev_unassigned = total_unassigned

    while -1 in assignments:
        # 检查时间限制
        if time.time() - start_time > max_time:
            print(f"警告：拍卖算法已达到时间限制 {max_time} 秒，提前终止")
            break
        
        # 检查恢复率
        current_recovery_rate = np.sum(assignments != -1) / n_cipher
        if current_recovery_rate >= recovery_threshold:
            print(f"拍卖算法达到恢复阈值 {recovery_threshold}，提前终止")
            break
            
        unassigned_bidders = np.where(assignments == -1)[0]
        for bidder in unassigned_bidders:
            # 计算每个卖家的净价值（负代价减去价格）
            net_values = -cost_matrix[bidder] - prices
            # 找到最大净价值和次大净价值
            sorted_indices = np.argsort(net_values)[::-1]
            max_value = net_values[sorted_indices[0]]
            second_max_value = net_values[sorted_indices[1]] if len(sorted_indices) > 1 else max_value
            # 计算投标价格
            bid = -cost_matrix[bidder, sorted_indices[0]] - (second_max_value - epsilon)
            # 更新卖家的价格
            prices[sorted_indices[0]] = bid
            # 检查是否有其他买家已经匹配到该卖家
            if sorted_indices[0] in assignments:
                previous_bidder = np.where(assignments == sorted_indices[0])[0][0]
                assignments[previous_bidder] = -1
            # 更新当前买家的匹配结果
            assignments[bidder] = sorted_indices[0]

        # 更新进度条
        current_unassigned = np.sum(assignments == -1)
        progress = prev_unassigned - current_unassigned
        if progress > 0:
            pbar.update(progress)
            prev_unassigned = current_unassigned

    pbar.close()
    return assignments

def AttackUsingAuxiliary(matrix_cipher, matrix_plain, selected_column_sse, dataset_name, max_time=300, recovery_threshold=0.9):
    startTime = time.time()

    # matrix_cipher = functions.read_csv_to_matrix(filePath)
    # selected_column_sse = ['Hospital','Pincipal Diagnosis']

    
    matrix_cipher_sse = functions.generate_submatrix(matrix_cipher, selected_column_sse)
    keyword_count = functions.count_keywords(matrix_cipher_sse[1:]) # 一共有多少个关键字
    matrix_plain_sse = functions.generate_submatrix(matrix_plain, selected_column_sse)

    # 每个关键字的volume
    volumn_cipher_sse = functions.count_frequency_multicol(np.array(matrix_cipher_sse[1:]), matrix_cipher_sse[0])
    volumn_plain_sse = functions.count_frequency_multicol(np.array(matrix_plain_sse[1:]), matrix_plain_sse[0])

    value_mapping = {}
    for key, dict1 in volumn_cipher_sse.items():
        dict2 = volumn_plain_sse[key]
        temp = functions.find_closest_mapping(dict1, dict2)
        value_mapping.update(temp)

    for column in selected_column_sse:
        plain = functions.extract_columns(matrix_plain, column)
        cipher = functions.extract_columns(matrix_cipher, column)
        keyword_list = list(set(plain).union(cipher))

        frquency_dict = {}
        frequency_folder = '/media/ices/machenrry/zl/Attack for DataBlinder/frequency/'
        for value in keyword_list:
            csv_file_path = os.path.join(frequency_folder, f'{value}.csv')
            if os.path.exists(csv_file_path):
                value_dict = functions.process_csv_file(csv_file_path)
                frquency_dict[value] = value_dict

        freq_cipher_dict = {key: value for key,value in frquency_dict.items() if key in cipher}
        freq_plain_dict = {key: value for key,value in frquency_dict.items() if key in plain}

        # Step 1 计算v = 匹配上的文档数/总文档数
        volume_cipher_dict = volumn_cipher_sse[column]
        volume_plain_dict = volumn_plain_sse[column]

        # 前者是每个时间间隔内的查询频率，后者是总的查询频率
        freq_cipher_dict_inPeriod, freq_cipher_dict_Total = functions.numToFreqency(freq_cipher_dict)
        freq_plain_dict_inPeriod, freq_plain_dict_Total = functions.numToFreqency(freq_plain_dict)

        feature_vector_cipher = compute_feature_vector(matrix_cipher, column, dataset_name)
        feature_vector_plain = compute_feature_vector(matrix_plain, column, dataset_name)
        
        cost_matrix = compute_cost_matrix(feature_vector_cipher, freq_cipher_dict_inPeriod, volume_cipher_dict, 
                                        feature_vector_plain, freq_plain_dict_inPeriod, volume_plain_dict)
        
        assignments = auction_algorithm(cost_matrix, max_time=max_time, recovery_threshold=recovery_threshold)

        sorted_cipher = sorted(freq_cipher_dict_inPeriod.keys())
        sorted_plain = sorted(freq_plain_dict_inPeriod.keys())

        # 输出匹配结果
        mapping = {}
        for i, assignment in enumerate(assignments):
            if assignment != -1:  # 只处理已匹配的结果
                mapping[sorted_cipher[i]] = sorted_plain[assignment]
        value_mapping.update(mapping)

        

    endTime = time.time()
    totalTime = round(endTime - startTime, 2)

    accuracy = 0
    count = 0
    for key, value in value_mapping.items():
        if key == value:
            count += 1

    accuracy = count / keyword_count
    return value_mapping, totalTime, accuracy

if __name__ == '__main__':
    filePathPlain = "F:/Desktop/Attack for Datablinder/2015/2015.csv"
    # quarters = ['4q2010']
    out = 'result/20250802/output of SSEAttack-ours.txt'
    matrix_plain = functions.read_csv_to_matrix(filePathPlain)

    root = "F:/Desktop/Attack for Datablinder/4q2010/text_508029.csv"
    base = [500, 725, 1050, 1525, 2210, 3205, 4645, 6735, 9765, 14160, 20530, 29770, 43170, 62600, 90750, 131600, 190850, 276750, 401300, 508029]
    matrix = functions.read_csv_to_matrix(root)


    with open(out, 'w', encoding='utf-8') as f:
        selected_column_sse = ['Hospital','Pincipal Diagnosis']
        for i in base:
            print(i)
            total_times = []
            accuracies = []
            for _ in range(10):
                matrix_cipher = functions.random_extract(matrix, i)
                mapping, totalTime, accuracy = AttackUsingAuxiliary(matrix_cipher, matrix_plain, selected_column_sse, "PUDF")
                total_times.append(totalTime)
                accuracies.append(accuracy)
                            # 计算平均值
            avg_time = sum(total_times) / 10
            avg_accuracy = sum(accuracies) / 10
            print(avg_accuracy)
            print("---------------------")
            # 将结果写入文件
            f.write(f"文件路径: 4q2010\\text {i}.txt, 执行时间: {avg_time}秒, 准确率: {avg_accuracy}\n")
            f.write("-" * 50 + "\n")
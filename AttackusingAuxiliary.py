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
            total_cost = 10 * fea_cost + freq_cost + vol_cost
            
            # 按照key的index存储总代价
            cost_matrix[cipher_idx, plain_idx] = total_cost
    return cost_matrix

def auction_algorithm(cost_matrix, epsilon=0.1):
    """
    拍卖算法实现

    :param cost_matrix: 代价矩阵，shape 为 (n_cipher, n_plain)
    :param epsilon: 投标增量，用于控制投标的步长
    :return: 匹配结果，数组中每个元素表示 sorted_cipher_keys 中对应关键字匹配到的 sorted_plain_keys 中的索引
    """
    n_cipher, n_plain = cost_matrix.shape
    prices = np.zeros(n_plain)  # 初始化卖家的价格
    assignments = np.full(n_cipher, -1)  # 初始化匹配结果，-1 表示未匹配

    # 计算总迭代次数用于进度条显示
    total_unassigned = np.sum(assignments == -1)
    pbar = tqdm(total=total_unassigned, desc='拍卖算法进行中')
    prev_unassigned = total_unassigned

    while -1 in assignments:
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

def AttackUsingAuxiliary(matrix_cipher, matrix_plain, selected_column_sse, dataset_name):
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
        frequency_folder = 'F:/Desktop/Supplementary experiments/frequency/'
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
        
        assignments = auction_algorithm(cost_matrix)

        sorted_cipher = sorted(freq_cipher_dict_inPeriod.keys())
        sorted_plain = sorted(freq_plain_dict_inPeriod.keys())

        # 输出匹配结果
        mapping = {}
        for i, assignment in enumerate(assignments):
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
    # root = "F:/Desktop/Attack for Datablinder/1q2010/text_538066.csv"
    # filePathPlain = "F:/Desktop/Attack for Datablinder/2015/2015.csv"
    # filePath = "F:/Desktop/Supplementary experiments/Crime_cipher.csv"
    # matrix_cipher = functions.read_csv_to_matrix(filePath)
    # filePathPlain = "F:/Desktop/Supplementary experiments/Crime_plain.csv"
    # matrix_plain = functions.read_csv_to_matrix(filePathPlain)
    # root = "F:/Desktop/Supplementary experiments/test.csv"
    # filePathPlain = "F:/Desktop/Supplementary experiments/train.csv"
    # selected_columns_sse = ["Crm_Cd_Desc"]
    # mapping, totalTime, accuracy = AttackUsingAuxiliary(matrix_cipher, matrix_plain, selected_columns_sse, "Crime")
    # print(accuracy)
    root = "F:/Desktop/Attack for Datablinder/"
    filePathPlain = "F:/Desktop/Attack for Datablinder/2015/2015.csv"
    quarters = ['4q2010']# '1q2010', '2q2010', '3q2010',
    out = 'result/output of SSEAttack-ours.txt'
    matrix_plain = functions.read_csv_to_matrix(filePathPlain)
    # 打开输出文件
    with open(out, 'w', encoding='utf-8') as f:
        for base in quarters:
            rootPath = os.path.join(root,base)
            file_list = os.listdir(rootPath)
            for file_name in file_list:
                filePath = os.path.join(rootPath,file_name)
                # filePathPlain = os.path.join(rootPath,file_name)
                matrix_cipher = functions.read_csv_to_matrix(filePath)
                name = os.path.join(base, file_name)
                selected_column_sse = ['Hospital','Pincipal Diagnosis']
                print(name)
                # 每个文件运行10次取平均值
                total_times = []
                accuracies = []
                for _ in range(10):
                    mapping, totalTime, accuracy = AttackUsingAuxiliary(matrix_cipher, matrix_plain, selected_column_sse, "PUDF")
                    total_times.append(totalTime)
                    accuracies.append(accuracy)
                
                # 计算平均值
                avg_time = sum(total_times) / 10
                avg_accuracy = sum(accuracies) / 10
                # 将结果写入文件
                f.write(f"文件路径: {name}, 执行时间: {totalTime}秒, 准确率: {accuracy}\n")
                f.write("-" * 50 + "\n")
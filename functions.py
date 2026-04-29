import csv
import random
import pandas as pd
import os
from collections import Counter
import math
import numpy as np
from itertools import product
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from datetime import datetime
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

# 根据表头选择特定列生成子矩阵
def generate_submatrix(matrix, header, columns):
    header_indices = [matrix[0].index(col) for col in columns]
    submatrix = []  
    for row in matrix[1:]:
        subrow = [row[i] for i in header_indices]
        submatrix.append(subrow)
    return submatrix

# 创建字典记录行号和对应的record id列的值
def create_rowid_dict(matrix, header, record_id_column):
    rowid_dict = {}
    record_id_index = header.index(record_id_column)
    for index, row in enumerate(matrix[1:], start=1):
        record_id = row[record_id_index]
        rowid_dict[record_id] = index
    return rowid_dict

# 读取CSV文件并生成矩阵
def read_csv_to_matrix(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            matrix.append(row)
    file.close()
    return matrix

def count_keywords(matrix):
    keyword_set = set()
    for row in matrix:
        for element in row:
            keyword_set.add(element)
    return len(keyword_set)

# 二元矩阵转十进制序列
def binary_to_decimal(matrix):
    # 调换矩阵的行列
    transposed_matrix = matrix.T
    # 将每一行的值组合起来转为十进制
    decimal_values = []
    for row in transposed_matrix:
        decimal_value = int("".join(map(str, row)), 2)
        decimal_values.append(decimal_value)
    return decimal_values

def find_unique_rows_withNone(matrix):
   # 使用字典来存储行数据及其出现次数
    row_count = {}
    unique_rows = []

    for row in matrix:
        serial_number = row[0]
        row_data = tuple(row[1:])

        # 检查当前行是否包含 None
        if None not in row_data:
            if row_data in row_count:
                row_count[row_data].append(serial_number)
            else:
                row_count[row_data] = [serial_number]

    # 找出仅出现一次的行
    for row_data, serial_numbers in row_count.items():
        if len(serial_numbers) == 1:
            unique_rows.extend(serial_numbers)

    return unique_rows

def find_unique_rows(matrix):
    # 使用字典来存储行的出现次数
    row_count = {}

    # 遍历每一行，从第二列开始将数据组合成一个元组
    for row in matrix:
        row_data = tuple(row[1:])  # 从第二列开始组合
        if row_data in row_count:
            row_count[row_data] += 1
        else:
            row_count[row_data] = 1

    # 找出仅出现一次的行
    unique_rows = [list(row) for row, count in row_count.items() if count == 1]
    return unique_rows

def replace_values_with_none(matrix, reference_dict):
    # 遍历每一行，从第二列开始检查
    for row in matrix:
        for i in range(1, len(row)):
            if row[i] not in reference_dict:
                row[i] = None
    return matrix

def generate_decimal_sequence(matrix):
    # 将矩阵转为DataFrame
    df = pd.DataFrame(matrix[1:], columns=matrix[0])  # 忽略第一行作为列名
    # 删除第一列（序列号），只保留关键字
    df_keywords = df.drop(columns=[df.columns[0]])
    # 为每个关键字生成二进制矩阵
    binary_matrix = pd.get_dummies(df_keywords.apply(lambda x: pd.Series(x)).stack()).groupby(level=0).sum()
    # 确保列名匹配关键字
    keywords = sorted(set(df_keywords.values.flatten()))
    binary_matrix = binary_matrix.reindex(columns=keywords, fill_value=0)
    # 将每一列转化为二进制字符串，并转化为十进制数
    decimal_sequence = [int(''.join(map(str, binary_matrix[keyword])), 2) for keyword in keywords]
    return decimal_sequence, keywords

def euclidean_distance(dict1, dict2):
    distance = 0
    if len(dict1) != len(dict2):
        all_keys = set(dict1.keys()).union(dict2.keys())  # 获取两个字典的所有键
        sorted_keys = sorted(all_keys)  # 对键进行排序

        
        values1 = [dict1.get(key, 0) for key in sorted_keys]  # 提取第一个字典的值，如果键不存在则填充为0
        values2 = [dict2.get(key, 0) for key in sorted_keys]  # 提取第二个字典的值，如果键不存在则填充为0

        # 计算这两个列表之间的欧氏距离
        distance = math.sqrt(sum((value1 - value2) ** 2 for value1, value2 in zip(values1, values2)))
    else:
        values1 = sorted(dict1.values())
        values2 = sorted(dict2.values())

        # 计算这两个列表之间的欧氏距离
        distance = math.sqrt(sum((value1 - value2) ** 2 for value1, value2 in zip(values1, values2)))

    return distance

def select_rows_and_generate_matrix(num_rows, csv_file):
    # 用于存储最终选择的行，包括表头
    selected_rows = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        # 将CSV文件的所有行读取到一个列表中
        all_rows = list(reader)
        header = all_rows[0]
        data_rows = all_rows[1:]
        
        # 检查请求的行数是否超过CSV文件中的总行数（不包括表头）
        if num_rows > len(data_rows):
            raise ValueError("请求的行数超过CSV文件中的总行数")
        selected_data_rows = random.sample(data_rows, num_rows)
        selected_rows.append(header)
        selected_rows.extend(selected_data_rows)
    return selected_rows

def write_dicts_to_txt(column_count, recovered_count, ope_count, det_count, output_file):
    with open(output_file, 'w') as file:
        for key in column_count.keys():
            line = f"{key}: {column_count[key]} columns need recovery, {recovered_count[key]} columns successfully recovered, OPE: {ope_count[key]}, DET: {det_count[key]}\n"
            file.write(line)

def generate_submatrix(matrix, columns):
    # 获取表头
    header = matrix[0]
    # 找到选定列的索引
    header_indices = [header.index(col) for col in columns]
    # 创建子矩阵并保留表头
    submatrix = [columns]  # 添加新的表头
    for row in matrix[1:]:
        subrow = [row[i] for i in header_indices]
        submatrix.append(subrow)
    return submatrix

def shuffle_matrix(matrix):
    df = pd.DataFrame(matrix)
    df.columns = matrix[0]
    new_matrix = df[1:].take(np.random.permutation(len(matrix[0])),axis=1).take(np.random.permutation(len(df[1:])),axis=0)
    new_matrix1 = new_matrix.values.tolist()
    new_matrix = [new_matrix.columns.values.tolist()] + new_matrix1
    return new_matrix

def count_column_elements(matrix):
    header = matrix[0]
    result = {}
    for col_idx in range(0, len(matrix[0])):
        col_name = header[col_idx]
        col_freq = {}
        if col_name == "Hospital" or col_name == "Pincipal Diagnosis":
            col_freq = {}
        else:
            for row in matrix[1:]:
                element = row[col_idx]
                if element in col_freq:
                    col_freq[element] += 1
                else:
                    col_freq[element] = 1
        result[col_name] = col_freq
    return result

def is_unique_value(dictionary, value):
    value_count = 0
    for val in dictionary.values():
        if val == value:
            value_count += 1
            if value_count > 1:
                return False
    return True

# 提取指定行数的子矩阵作为敌手拿到的明文信息
def random_submatrix(matrix, num_rows):
    header = matrix[0]
    data = matrix[1:]
    
    random_rows = random.sample(data, num_rows)
    submatrix = [header] + random_rows
    
    return submatrix

def convert_nested_counts_to_frequencies(nested_count_dict):
    frequency_dict = {}
    
    for col_name, sub_dict in nested_count_dict.items():
        total_count = sum(sub_dict.values())
        frequency_dict[col_name] = {key: value / total_count for key, value in sub_dict.items()}
    
    return frequency_dict

def count_frequency_multicol(matrix, column_names): 
    num_cols = matrix.shape[1]
    frequency_dicts = [{} for _ in range(num_cols)]

    for col_idx in range(num_cols):
        column = matrix[:, col_idx]
        total_count = len(column)

        for key in column:
            key_str = str(key)
            frequency_dicts[col_idx][key_str] = frequency_dicts[col_idx].get(key_str, 0) + 1 / total_count

    # Create the final dictionary with column names as keys
    final_dict = {}

    for col_idx, freq_dict in enumerate(frequency_dicts):
        col_name = column_names[col_idx]
        final_dict[col_name] = freq_dict

    return final_dict

def convert_to_date(year_month):
    year, month = map(int, year_month.split('.'))
    return datetime(year, month, 1)

# 给定的数据列表
data = ['2009.01-2009.12', '2010.01-2010.12', '2011.01-2011.12', '2012.01-2012.12',
        '2013.01-2013.12', '2014.01-2014.12', '2015.01-2015.12',
        '2016.01-2016.12', '2017.01-2017.12', '2018.01-2018.12',
        '2019.01-2019.12', '2020.01-2020.12', '2021.01-2021.12',
        '2022.01-2022.12', '2023.01-2023.12', '2024.01-2024.12']

# 判断给定的 (year).(month) 在哪一个区间
def find_period(year_month):
    target_date = convert_to_date(year_month)
    
    for idx, period in enumerate(data):
        start_year_month, end_year_month = period.split('-')
        start_date = convert_to_date(start_year_month)
        end_date = convert_to_date(end_year_month)
        
        if start_date <= target_date <= end_date:
            return period
        
# 读取CSV文件中的内容，生成字典
def process_csv_file(file_path):
    result_dict = {}
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            time_period = row[0]
            frequency = int(row[1])  # 假设频率为浮点数
            times = time_period.split("-")
            year = times[0]
            month = times[1]
            t = f'{year}.{month}'
            key = find_period(t)
            if key in result_dict:
                result_dict[key] += frequency
            else:
                result_dict[key] = frequency
    return result_dict

def numToFreqency(data_dict):
    # 计算每个子字典中时间段的频数总和
    # try:
    #     total_counts = {key: sum(sub_dict.values()) for key, sub_dict in data_dict.items()}
    # except:
    #     print(data_dict)
    total_counts = {key: sum(sub_dict.values()) for key, sub_dict in data_dict.items()}
    # 计算整个字典的总频数
    total_frequency = sum(total_counts.values())
    data_dict1 = {}
    data_dict2 = {}
    # 更新每个子字典中的值为频率
    for key, sub_dict in data_dict.items():
        freq = 0
        if None in sub_dict.keys():
            sub_dict.pop(None)
        if not bool(sub_dict):
            continue
        else:
            # 获取所有时间间隔并排序
            sorted_periods = sorted(sub_dict.keys())
            # 创建按时间顺序排列的频率列表
            freq_list = []
            for period in sorted_periods:
                freq_list.append(sub_dict[period] / total_frequency)
                freq += sub_dict[period]
            data_dict1[key] = freq_list
            data_dict2[key] = freq / total_frequency
    return data_dict1, data_dict2

def all_numeric(input_list):
    return all(item.isdigit() for item in input_list)

def custom_sort(value):
    # 尝试将字符串转换为数字
    try:
        return int(value)  # 如果是纯数字，返回整数
    except ValueError:
        return value  # 否则返回原字符串

def are_pairs_equal(pair1, pair2):
    return frozenset(pair1) == frozenset(pair2)

def calculate_keyword_cooccurrence_frequency(keyword_docs, list1, list2):
    # 构建反向索引：文档 ID -> 关键字集合
    doc_to_keywords = defaultdict(set)

    # 构建反向索引
    for keyword, docs in keyword_docs.items():
        for doc in docs:
            doc_to_keywords[doc].add(keyword)

    # 初始化计数字典，用于统计每对关键字出现的次数
    keyword_pair_counts = defaultdict(int)
    total_combinations = 0

    # 遍历每个文档的关键字集合，使用 tqdm 添加进度条
    for keywords in tqdm(doc_to_keywords.values(), desc="Processing documents", total=len(doc_to_keywords)):
        # 获取两个列表中关键字的交集
        keys_in_list1 = set(list1).intersection(keywords)
        keys_in_list2 = set(list2).intersection(keywords)

        # 计算所有组合
        if keys_in_list1 and keys_in_list2:
            # 笛卡尔积计算组合数
            combinations = list(product(keys_in_list1, keys_in_list2))
            common_count = len(combinations)
            total_count = len(keys_in_list1) * len(keys_in_list2)
            
            for key1, key2 in combinations:
                keyword_pair_counts[(key1, key2)] += 1
            total_combinations += total_count

    # 计算每对关键字的出现频率
    frequency_dict = {pair: count / total_combinations for pair, count in keyword_pair_counts.items()}

    return frequency_dict

def isTwoDictEqual(dict1, dict2):
    result = {}
    for key1, value1 in dict1.items():
        for key2, value2 in dict2.items():
            if abs(value1-value2) < 0.001:
                result[key1] = key2
    return result

def extract_columns(matrix, target_column):
    # 找出目标列的索引
    headers = matrix[0]
    index = headers.index(target_column)
    # 提取目标列的数据
    return [row[index] for row in matrix[1:]]

def compute_frequency_and_cdf(data):
    if all_numeric(data):
        data1 = [int(i) for i in data]
        freq = Counter(data1)
    else:
        freq = Counter(data)
    total = sum(freq.values())
    cdf = {}
    cumulative = 0

    for key in sorted(freq):
        count = freq[key]
        cumulative += count
        freq[key] /= total
        cdf[key] = cumulative / total

    return freq, cdf

def find_optimal_mapping(data_c, data_z):
    freq_c, cdf_c = compute_frequency_and_cdf(data_c)
    freq_z, cdf_z = compute_frequency_and_cdf(data_z)

    keys_c = sorted(freq_c.keys())
    keys_z = sorted(freq_z.keys())

    cost_matrix = np.zeros((len(keys_c), len(keys_z)))

    for i, key_c in enumerate(keys_c):
        for j, key_z in enumerate(keys_z):
            freq_diff = (freq_c[key_c] - freq_z[key_z]) ** 2
            cdf_diff = (cdf_c[key_c] - cdf_z[key_z]) ** 2
            cost_matrix[i, j] = freq_diff + cdf_diff

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    mapping = {keys_c[i]: keys_z[j] for i, j in zip(row_ind, col_ind)}
    return mapping

def compute_frequency_and_cdf_A4(data):
    # if all_numeric(data):
    #     data1 = [int(i) for i in data]
    #     freq = Counter(data1)
    # else:
        # freq = Counter(data)
    freq = Counter(data)
    total = sum(freq.values())
    cdf = {}
    cumulative = 0

    for key in sorted(freq):
        count = freq[key]
        cumulative += count
        # freq[key] /= total
        # cdf[key] = cumulative / total
        freq[key] = (count / total)  # 将频率扩大10000倍
        cdf[key] = (cumulative / total)  # 将CDF扩大10000倍

    return freq, cdf

def find_optimal_mapping_A4(data_c, data_z):
    freq_c, cdf_c = compute_frequency_and_cdf_A4(data_c)
    freq_z, cdf_z = compute_frequency_and_cdf_A4(data_z)

    keys_c = sorted(freq_c.keys())
    keys_z = sorted(freq_z.keys())

    cost_matrix = np.zeros((len(keys_c), len(keys_z)))

    for i, key_c in enumerate(keys_c):
        for j, key_z in enumerate(keys_z):
            freq_diff = (freq_c[key_c] - freq_z[key_z]) ** 2
            cdf_diff = (cdf_c[key_c] - cdf_z[key_z]) ** 2
            cost_matrix[i, j] = freq_diff + cdf_diff

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    mapping = {keys_c[i]: keys_z[j] for i, j in zip(row_ind, col_ind)}
    return mapping

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

def random_extract(matrix, num_rows):
    """
    从矩阵中随机抽取指定行数，保留表头
    
    参数:
        matrix: 原始矩阵，list形式，第一行为表头
        num_rows: 需要抽取的行数（不包含表头）
    
    返回:
        新矩阵，包含表头和随机抽取的行
    """
    if len(matrix) <= 1:
        return matrix  # 只有表头或空矩阵，直接返回
    
    # 确保抽取的行数不超过可用数据行
    data_rows = matrix[1:]  # 所有数据行（排除表头）
    actual_num = min(num_rows, len(data_rows))
    
    # 随机抽取指定数量的行
    selected_rows = random.sample(data_rows, actual_num)
    
    # 组合表头和抽取的行，形成新矩阵
    return [matrix[0]] + selected_rows

def column_frequencies(matrix):
    """
    高效计算矩阵中每一列的元素出现频率
    
    参数:
        matrix: 2D numpy array 或 嵌套列表
    返回:
        字典，键为列索引，值为该列元素出现频率的字典
    """
    # 直接处理列表输入，避免重复numpy转换
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    
    # 提前获取矩阵形状
    rows, cols = matrix.shape
    frequencies = {}
    
    # 预分配列数组
    col_array = np.empty(rows, dtype=matrix.dtype)
    
    for col_idx in range(cols):
        # 直接使用numpy的列视图，避免复制
        col_data = matrix[:, col_idx]
        
        # 使用高效的numpy unique函数
        unique, counts = np.unique(col_data, return_counts=True)
        total = rows
        
        # 使用向量化操作计算频率
        freq = counts / total
        
        # 构建频率字典，保留5位小数
        # 根据数据类型选择键的存储方式
        if matrix.dtype.kind in 'iufc':  # 数值类型
            freq_dict = {key: round(val, 5) for key, val in zip(unique, freq)}
        else:  # 其他类型转换为字符串
            freq_dict = {str(key): round(val, 5) for key, val in zip(unique, freq)}
        
        frequencies[col_idx] = freq_dict
    
    return frequencies

# 计算是否存在特殊的行（将一行变成一个字符串，然后看独特性）
def get_unique_values(my_list):
    unique_values = set()
    duplicate_values = set()

    for item in my_list:
        if item not in duplicate_values:
            if item in unique_values:
                unique_values.remove(item)
                duplicate_values.add(item)
            else:
                unique_values.add(item)

    return list(unique_values)

# 找到字典2中和自己最接近的key
def find_closest_mapping(dict1, dict2):
    mapping = {}
    for key1, value1 in dict1.items():
        min_distance = np.inf
        candidate = key1
        for key2, value2 in dict2.items():
            distance = abs(value1 - value2)
            if distance < min_distance:
                min_distance = distance
                candidate = key2
        mapping[key1] = candidate
    return mapping

def find_unique_value(input_list):
    value_counts = {}
    
    for value in input_list:
        if value in value_counts:
            value_counts[value] += 1
        else:
            value_counts[value] = 1
    
    unique_values = [key for key, count in value_counts.items() if count == 1]
    return unique_values

def replace_nested_inplace(lst, old_value, new_value):
    for i in range(len(lst)):
        if isinstance(lst[i], list):  # 如果是子列表，递归处理
            replace_nested_inplace(lst[i], old_value, new_value)
        else:  # 否则直接替换
            if lst[i] == old_value:
                lst[i] = new_value
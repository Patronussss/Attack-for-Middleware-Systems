import csv
from tqdm import tqdm
import pandas as pd
import os 
import re
import random
import numpy as np
from collections import Counter
import math 
from itertools import product
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from datetime import datetime
import os


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
        sub_dict1 = {}
        freq = 0
        if not bool(sub_dict):
            continue
        else:
            for period, count in sub_dict.items():
                sub_dict1[period] = count / total_frequency
                freq += count
            data_dict1[key] = sub_dict1
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

# 读取CSV文件并生成矩阵
def read_csv_to_matrix(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            matrix.append(row)
    return matrix

# 提取指定行数的子矩阵作为敌手拿到的明文信息
def random_submatrix(matrix, num_rows):
    header = matrix[0]
    data = matrix[1:]
    
    random_rows = random.sample(data, num_rows)
    submatrix = [header] + random_rows
    
    return submatrix

# 定义一个函数，用于统计矩阵中出现的关键字数量
def count_keywords(matrix):
    keyword_set = set()
    for row in matrix:
        for element in row:
            keyword_set.add(element)
    return len(keyword_set)

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

def column_frequencies(matrix):
    """
    计算矩阵中每一列的元素出现频率，并将结果存储到字典中。
    
    :param matrix: 2D numpy array or list of lists
    :return: 字典，键为列索引，值为该列元素出现频率的字典
    """
    # 将输入数据转换为numpy数组
    matrix = np.array(matrix)
    # 初始化结果字典
    frequencies = {}
    
    # 遍历矩阵的每一列
    for col_index in range(matrix.shape[1]):
        col = matrix[:, col_index]  # 提取列数据
        unique, counts = np.unique(col, return_counts=True)  # 计算唯一值及其计数
        total = len(col)  # 该列的总元素数量
        # 计算频率并保留5位小数
        freq_dict = {str(key): round(count / total, 5) for key, count in zip(unique, counts)}
        frequencies[col_index] = freq_dict  # 将频率字典存储到列索引对应的位置
    
    return frequencies

def replace_values_with_none(matrix, reference_dict):
    # 遍历每一行，从第二列开始检查
    for row in matrix:
        for i in range(1, len(row)):
            if row[i] not in reference_dict:
                row[i] = None
    return matrix

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

index_list = [1]
frac_list = [0.1,0.3,0.5,0.7,0.9]

for frac in frac_list:
    for il in index_list:
        base = 'F:/Desktop/text/'+ str(il)+ 'q2010/'
        # target = '/Users/cherry/Desktop/zengli/'+ str(il)+ 'q2010plain/'
        out = 'F:/Desktop/text/20250116/A3/2/'+ str(il)+ 'q2010 output of A3/' + str(frac) + '/'
        if not os.path.exists(out):
                os.makedirs(out)
        # lists = [i[:-4] for i in os.listdir(out)]
        file_list = os.listdir(base)
        for file_name in sorted(file_list):
            kw_c = 0 # 关键字总数
            kw_s_c = 0 # 恢复的关键字数
            re_c = 0 # 记录总数
            re_s_c = 0 # 恢复的记录数
            times = 0
            while times < 5:

                print("---------------------------")
                print(file_name)
                print(frac)
                print("---------------------------")
                filePath = os.path.join(base,file_name)
                matrix = read_csv_to_matrix(filePath)

                selected_columns_cipher = ['Record ID', 'Age', 'Admission Type', 'Length of stay', 'Risk','Discharge', 'Gender', 'Race','Hospital', 'Pincipal Diagnosis']

                matrix_cipher = generate_submatrix(matrix, selected_columns_cipher)
                num_rows_to_extract =  int((len(matrix_cipher)-1) * frac)

                matrix_plain = random_submatrix(matrix_cipher, num_rows_to_extract)
                record_count = len(matrix_plain)-1 #能够过恢复的最多的record 记录（剩下的没有对照的identifire就算恢复了所有数据也没法比较）
                keyword_count = count_keywords(generate_submatrix(matrix_cipher,['Age', 'Admission Type', 'Length of stay', 'Risk','Discharge', 'Gender', 'Race','Hospital', 'Pincipal Diagnosis'])[1:]) #能够恢复的关键字个数，由于只能拿到部分明文数据库，因此能够拿到的关键字个数和明文数据库拥有的明文个数相关

                # 针对OPE的攻击，排序和CDF
                # 同时比较CDF和频率，找到一一对应关系
                selected_column_ope = ['Record ID', 'Age', 'Admission Type', 'Length of stay', 'Risk']
                matrix_cipher_ope = generate_submatrix(matrix_cipher, selected_column_ope)
                matrix_plain_ope = generate_submatrix(matrix_plain, selected_column_ope)

                # freq_cipher_age, cdf_cipher_age = calculate_frequency_and_cdf(matrix_cipher_ope,'Age')
                # freq_plain_age, cdf_plain_age = calculate_frequency_and_cdf(matrix_plain_ope, 'Age')

                # Age
                column_cipher_age = extract_columns(matrix_cipher_ope, 'Age')
                age_cipher_count = Counter(column_cipher_age)
                keyword_count_cipher_age = len(age_cipher_count)

                column_plain_age = extract_columns(matrix_plain_ope, 'Age')
                age_plain_count = Counter(column_plain_age)
                keyword_count_plain_age = len(age_plain_count)
                value_mapping = {}

                if keyword_count_cipher_age == keyword_count_plain_age:
                    sorted_cipher_age = sorted(set(column_cipher_age))
                    sorted_plain_age = sorted(set(column_plain_age))
                    mapping_ope_age = {key: value for key, value in zip(sorted_cipher_age, sorted_plain_age)}
                    value_mapping.update(mapping_ope_age)
                else:
                    mapping_ope_age = find_optimal_mapping(column_cipher_age, column_plain_age)
                    value_mapping.update(mapping_ope_age)

                # Admission Type
                column_cipher_admi = extract_columns(matrix_cipher_ope, 'Admission Type')
                admi_cipher_count = Counter(column_cipher_admi)
                keyword_count_cipher_admi = len(admi_cipher_count)

                column_plain_admi = extract_columns(matrix_plain_ope, 'Admission Type')
                admi_plain_count = Counter(column_plain_admi)
                keyword_count_plain_admi = len(admi_plain_count)

                if keyword_count_cipher_admi == keyword_count_plain_admi:
                    sorted_cipher_admi = sorted(set(column_cipher_admi))
                    sorted_plain_admi = sorted(set(column_plain_admi))
                    mapping_ope_admi = {key: value for key, value in zip(sorted_cipher_admi, sorted_plain_admi)}
                    value_mapping.update(mapping_ope_admi)
                else:
                    mapping_ope_admi = find_optimal_mapping(column_cipher_admi, column_plain_admi)
                    value_mapping.update(mapping_ope_admi)

                # Risk
                column_cipher_risk = extract_columns(matrix_cipher_ope, 'Risk')
                risk_cipher_count = Counter(column_cipher_risk)
                keyword_count_cipher_risk = len(risk_cipher_count)

                column_plain_risk = extract_columns(matrix_plain_ope, 'Risk')
                risk_plain_count = Counter(column_plain_risk)
                keyword_count_plain_risk = len(risk_plain_count)

                if keyword_count_cipher_risk == keyword_count_plain_risk:
                    sorted_cipher_risk = sorted(set(column_cipher_risk))
                    sorted_plain_risk = sorted(set(column_plain_risk))
                    mapping_ope_risk = {key: value for key, value in zip(sorted_cipher_risk, sorted_plain_risk)}
                    value_mapping.update(mapping_ope_risk)
                else:
                    mapping_ope_risk = find_optimal_mapping(column_cipher_risk, column_plain_risk)
                    value_mapping.update(mapping_ope_risk)

                # Length of Stay
                column_cipher_stay = extract_columns(matrix_cipher_ope, 'Length of stay')
                stay_cipher_count = Counter(column_cipher_stay)
                keyword_count_cipher_stay = len(stay_cipher_count)

                column_plain_stay = extract_columns(matrix_plain_ope, 'Length of stay')
                stay_plain_count = Counter(column_plain_stay)
                keyword_count_plain_stay = len(stay_plain_count)

                if keyword_count_cipher_stay == keyword_count_plain_stay:
                    sorted_cipher_stay = sorted(set(column_cipher_stay))
                    sorted_plain_stay = sorted(set(column_plain_stay))
                    mapping_ope_stay = {key: value for key, value in zip(sorted_cipher_stay, sorted_plain_stay)}
                    value_mapping.update(mapping_ope_stay)
                else:
                    sorted_cipher_stay = sorted(column_cipher_stay, key=custom_sort)
                    sorted_plain_stay = sorted(column_plain_stay, key=custom_sort)
                    mapping_ope_stay = find_optimal_mapping(sorted_cipher_stay, sorted_plain_stay)
                    value_mapping.update(mapping_ope_stay)

                # print(value_mapping)
                keyword_count_ope_det = count_keywords(generate_submatrix(matrix_plain,[
                    'Length of stay'
                    ,'Age', 'Admission Type', 'Risk'
                    ,'Discharge', 'Gender', 'Race'
                    ])[1:]) 
                print(f"OPE+DET一共包含{keyword_count_ope_det}个关键字")

                keyword_count_ope = count_keywords(generate_submatrix(matrix_plain,[
                    'Length of stay'
                    ,'Age', 'Admission Type', 'Risk'
                    # ,'Discharge', 'Gender', 'Race'
                    ])[1:]) 
                print(f"OPE一共包含{keyword_count_ope}个关键字")

                keyword_count_ope_nostay = count_keywords(generate_submatrix(matrix_plain,[
                    # 'Length of stay',
                    'Age', 'Admission Type', 'Risk',
                    'Discharge', 'Gender', 'Race'
                    ])[1:]) 
                print(f"其中不包含住院天数，剩下的关键字的个数为{keyword_count_ope_nostay}")
                # print(keyword_count)
                count = 0
                for key, value in value_mapping.items():
                    if key == value:
                        count += 1
                print(f"经过OPE后恢复了{count}个关键字")
                # print(count/keyword_count_ope)

                # DET部分
                selected_column_det = ['Discharge', 'Gender', 'Race']
                matrix_cipher_det = generate_submatrix(matrix_cipher, selected_column_det)
                matrix_plain_det = generate_submatrix(matrix_plain, selected_column_det)

                # 计算频率
                element_cipher_det = column_frequencies(matrix_cipher_det[1:])
                element_plain_det = column_frequencies(matrix_plain_det[1:])

                result = {}
                for key, dict1 in element_cipher_det.items():
                    dict2 = element_plain_det[key]
                    temp = find_closest_mapping(dict1, dict2)
                    value_mapping.update(temp)
                    result.update(temp)

                count = 0
                value_mapping_od = {}
                for key, value in value_mapping.items():
                    if key == value:
                        count += 1
                        value_mapping_od[key] = value
                
                # print(value_mapping)
                print(f"DET的关键字个数为：{len(result)}")
                print(f"经过OPE+DET恢复了{count}个关键字")

                # 先根据OPE 和 DET恢复一部分record id
                
                selected_columns_od = ['Record ID', 'Age', 'Admission Type', 'Risk','Discharge', 'Gender', 'Race']
                matrix_plain_od = generate_submatrix(matrix_plain, selected_columns_od)
                matrix_cipher_od = generate_submatrix(matrix_cipher, selected_columns_od)
                matrix_recovered_od = replace_values_with_none(matrix_cipher_od, value_mapping_od)
                list_cipher_od = []
                record_cipher = []
                for row in matrix_recovered_od[1:]:
                    if None not in row[1:]:
                        res = ' '.join(row[1:])
                        record_cipher.append(row[0])
                        list_cipher_od.append(res)

                list_plain_od = []
                record_plain = []
                for row in matrix_plain_od[1:]:
                    res = ' '.join(row[1:])
                    record_plain.append(row[0])
                    list_plain_od.append(res)

                def find_unique_value(input_list):
                    value_counts = {}
                    
                    for value in input_list:
                        if value in value_counts:
                            value_counts[value] += 1
                        else:
                            value_counts[value] = 1
                    
                    unique_values = [key for key, count in value_counts.items() if count == 1]
                    return unique_values

                unique_value_cipher_od = find_unique_value(list_cipher_od)
                unique_value_plain_od = find_unique_value(list_plain_od)

                unique_rows = [v for v in unique_value_plain_od if v in unique_value_cipher_od]

                recovered_rows = {}
                for row in unique_rows:
                    id_cipher = record_cipher[list_cipher_od.index(row)]
                    id_plain = record_plain[list_plain_od.index(row)]
                    recovered_rows[id_cipher] = id_plain

                print(len(recovered_rows))
                count = 0 
                for key,value in recovered_rows.items():
                    if key == value:
                        count += 1
                print(count)

                def create_rowid_dict(matrix, header, record_id_column):
                    rowid_dict = {}
                    record_id_index = header.index(record_id_column)
                    
                    for index, row in enumerate(matrix[1:], start=1):
                        record_id = row[record_id_index]
                        rowid_dict[record_id] = index
                    
                    return rowid_dict

                recordid_cipher = create_rowid_dict(matrix_cipher, matrix_cipher[0], 'Record ID')
                recordid_plain = create_rowid_dict(matrix_plain, matrix_plain[0], 'Record ID')

                for id_c, id_p in recovered_rows.items():
                    # index_cipher = recordid_cipher[id_c]
                    row_cipher = matrix_cipher[recordid_cipher[id_c]]
                    index_plain = recordid_plain[id_p]
                    row_plain = matrix_plain[index_plain]
                    for i in range(1, len(row_cipher)):
                        value_mapping_od[row_cipher[i]] = row_plain[i]
                                

                selected_columns_cipher_iod = ['Record ID', 'Age', 'Admission Type', 'Length of stay', 'Risk','Discharge', 'Gender', 'Race']

                matrix_cipher_iod = generate_submatrix(matrix_cipher, selected_columns_cipher_iod)  # 经过OPE和DET后恢复了多少个记录
                matrix_recovered = replace_values_with_none(matrix_cipher_iod, value_mapping_od)



                record_recovered = find_unique_rows_withNone(matrix_recovered)
                print(f"经过OPE+DET一共找到了{len(record_recovered)}条数据")

                # SSE
                selected_column_sse = ['Record ID', 'Hospital', 'Pincipal Diagnosis']

                matrix_cipher_sse = generate_submatrix(matrix_cipher, selected_column_sse)
                matrix_plain_sse = generate_submatrix(matrix_plain, selected_column_sse)

                matrix_recovered_sse = []
                record_recovered_sse = []

                total_iterations = len(matrix_plain_sse)
                with tqdm(total=total_iterations, desc="generating matrix recovered") as pbar:
                    for row in matrix_plain_sse:
                        if row[0] in record_recovered:
                            matrix_recovered_sse.append(row)
                            record_recovered_sse.append(row[0])
                        pbar.update(1)
                if len(matrix_recovered_sse) > 0:
                    # 生成两个十进制序列及其对应的关键字
                    decimal_sequence, keywords = generate_decimal_sequence(matrix_recovered_sse)
                    # 创建一个映射，将十进制值映射回关键字
                    decimal_to_keyword = {decimal: keyword for decimal, keyword in zip(decimal_sequence, keywords)}
                    # 找出每个序列中独特的值（只出现一次的值）
                    unique_values = [decimal for decimal in decimal_sequence if decimal_sequence.count(decimal) == 1]
                    # 找出两个序列中相同的独特值
                    common_unique_values = set(unique_values) & set(unique_values)

                    # 记录这些相同的值对应的关键字
                    common_keywords = {decimal_to_keyword[value]: decimal_to_keyword[value] for value in common_unique_values if value in decimal_to_keyword and value in decimal_to_keyword}
                    value_mapping_od.update(common_keywords)
                    # print(len(value_mapping))

                    keyword_count_sse = count_keywords(generate_submatrix(matrix_plain,[
                        'Length of stay',
                        'Age', 'Admission Type', 'Risk',
                        'Discharge', 'Gender', 'Race',
                        'Hospital', 'Pincipal Diagnosis'
                        ])[1:]) 
                    print(f"OPE+DET+SSE中一共包含{keyword_count_sse}个关键字")

                    # 返回到矩阵中看能不能获得更多
                    print(f"经过volume一共找到了{len(common_keywords)}个关键字")

                count = 0 
                value_mapping_ods = {}
                for key, value in value_mapping_od.items():
                    if key == value:
                        count += 1
                        value_mapping_ods[key] = value
                print(f"经过OPE+DET+volume一共找到了{count}个关键字")



                selected_columns_cipher_iods = ['Record ID', 'Age', 'Admission Type', 'Length of stay', 'Risk','Discharge', 'Gender', 'Race', 'Hospital', 'Pincipal Diagnosis']

                matrix_cipher_iods = generate_submatrix(matrix_cipher, selected_columns_cipher_iods)  # 经过OPE和DET后恢复了多少个记录
                matrix_recovered_sse = replace_values_with_none(matrix_cipher_iods, value_mapping_ods)

                record_recovered_iods = find_unique_rows_withNone(matrix_recovered_sse)

                # 创建字典记录行号和对应的record id列的值
                record_plain = [row[0] for row in matrix_plain]
                # print(matrix_plain[0])
                # record_plain = create_rowid_dict(matrix_plain, matrix_plain[0], 'Record ID')
                record = list(set(record_plain) & set(record_recovered_iods))

                if len(record) == 0 and len(matrix_plain) > 5000:
                    print("len(record) = 0")
                    continue

                print(f"成功恢复了{len(record)}条数据")
                print(f"一共拥有{len(matrix_plain)}条记录")
                print(f"恢复记录比例：{len(record) / len(matrix_plain)}")

                # matrix_plain_iods = generate_submatrix(matrix_plain, selected_columns_cipher_iods)

                # 使用字典来存储行的出现次数
                row_count = {}
                # 遍历每一行，从第二列开始将数据组合成一个元组
                for row in matrix_plain:
                    row_data = tuple(row[1:])  # 从第二列开始组合
                    if row_data in row_count:
                        row_count[row_data] += 1
                    else:
                        row_count[row_data] = 1

                # 找出仅出现一次的行
                unique_rows = [list(row) for row, count in row_count.items() if count == 1]
                print(f"plain中一共有{len(unique_rows)}条数据是独特的，也就是说理论上可以恢复的数据大小")

                # # 找到了拥有对应关系的行数
                # data_dict_cipher = {}
                # data_dict_plain = {}
                # stop_words = ["and", "of", "or", "for", "with", "to", "not", "by", "in", "the", "but", "from", "as"]

                # for row_num, row in enumerate(matrix_cipher_sse[1:]):
                #     for i in range(len(row)):
                #         word = row[i]
                #         if word.lower() not in stop_words:
                #             if word in data_dict_cipher:
                #                 data_dict_cipher[word].append(row_num)
                #             else:
                #                 data_dict_cipher[word] = [row_num]

                # for row_num, row in enumerate(matrix_plain_sse[1:]):
                #     for i in range(len(row)):
                #         word = row[i]
                #         if word.lower() not in stop_words:
                #             if word in data_dict_plain:
                #                 data_dict_plain[word].append(row_num)
                #             else:
                #                 data_dict_plain[word] = [row_num]

                # # 计算共现
                # keyword_unrecovered = [key for key in data_dict_cipher if key not in value_mapping_ods]
                # keyword_recovered = [key for key in data_dict_cipher if key in value_mapping_ods]
                # print(f"还有{len(keyword_unrecovered)}个关键字没有恢复")
                # keyw_un = [key for key in data_dict_plain if key not in value_mapping_ods]
                # keyw_re = [key for key in data_dict_plain if key in value_mapping_ods]
                # print(f"理论上还能恢复{len(keyw_un)}个关键字")
                # print(len(keyword_recovered))

                # frequency_dict_cipher = calculate_keyword_cooccurrence_frequency(data_dict_cipher, keyword_unrecovered, keyword_recovered)
                # frequency_dict_plain = calculate_keyword_cooccurrence_frequency(data_dict_plain, keyw_re, keyw_un)

                # new_dict = {}

                # # 获取 dict1 和 dict2 的键值对列表
                # items1 = list(frequency_dict_cipher.items())
                # items2 = list(frequency_dict_plain.items())

                # # 使用 itertools.product 比较每对值
                # for (key1, value1), (key2, value2) in product(items1, items2):
                #     if abs(value1 - value2) < 0.00009:
                #         new_dict[key1] = key2
                # print(len(new_dict))
                # count = 0
                # for key, value in new_dict.items():
                #     if are_pairs_equal(key, value):
                #         count += 1
                #         value_mapping_ods[key] = value
                # print(count)
                # print(len(value_mapping_ods))

                # matrix_recovered_sse = replace_values_with_none(matrix_cipher_iods, value_mapping_ods)

                # record_recovered_iods = find_unique_rows_withNone(matrix_recovered_sse)

                # record_plain = [row[0] for row in matrix_plain]
                # # print(matrix_plain[0])
                # # record_plain = create_rowid_dict(matrix_plain, matrix_plain[0], 'Record ID')
                # record_final = list(set(record_plain) & set(record_recovered_iods))   
                # SSE
                # Step 1：尝试使用volume leakage，即寻找每一列中匹配记录频率独特的关键字
                selected_column_sse = ['Hospital','Pincipal Diagnosis']
                matrix_cipher_sse = generate_submatrix(matrix_cipher, selected_column_sse)
                matrix_plain_sse = generate_submatrix(matrix_plain, selected_column_sse)

                # 每个关键字的volume
                volumn_cipher_sse = count_frequency_multicol(np.array(matrix_cipher_sse[1:]), matrix_cipher_sse[0])
                volumn_plain_sse = count_frequency_multicol(np.array(matrix_plain_sse[1:]), matrix_plain_sse[0])
                mapping_sse = {}
                for key, dict1 in volumn_cipher_sse.items():
                    dict2 = volumn_plain_sse[key]
                    temp = find_closest_mapping(dict1, dict2)
                    value_mapping.update(temp)
                    mapping_sse.update(temp)

                for key, value in mapping_sse.items():
                    if key == value:
                        value_mapping_od[key] = value
                    
                # keyword_number = count_keywords(generate_submatrix(matrix_cipher, selected_columns_noid)[1:])

                hospital_plain = extract_columns(matrix_plain, 'Hospital')
                hospital_cipher = extract_columns(matrix_cipher, 'Hospital')
                hospital_list = list(set(hospital_plain).union(set(hospital_cipher)))
                diagnosis_plain = extract_columns(matrix_plain, 'Pincipal Diagnosis')
                diagnosis_cipher = extract_columns(matrix_cipher, 'Pincipal Diagnosis')

                diagnosis_list = list(set(diagnosis_plain).union(set(diagnosis_cipher)))

                list_dh = hospital_list + diagnosis_list

                hospital_frequency_dict = {}

                frequency_folder = 'frequency/'
                for value in hospital_list:
                    csv_file_path = os.path.join(frequency_folder, f'{value}.csv')
                    if os.path.exists(csv_file_path):
                        value_dict = process_csv_file(csv_file_path)
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
                        value_dict = process_csv_file(csv_file_path)
                        diagnosis_frequency_dict[value] = value_dict

                # 每个关键字的搜索频率
                diag_freq_cipher_dict = {key: value for key,value in diagnosis_frequency_dict.items() if key in diagnosis_cipher}
                diag_freq_plain_dict = {key: value for key,value in diagnosis_frequency_dict.items() if key in diagnosis_plain}
                # Step 1 计算v = 匹配上的文档数/总文档数
                hos_volume_cipher_dict = volumn_cipher_sse['Hospital']
                diag_volume_cipher_dict = volumn_cipher_sse['Pincipal Diagnosis']
                hos_volume_plain_dict = volumn_plain_sse['Hospital']
                diag_volume_plain_dict = volumn_plain_sse['Pincipal Diagnosis']

                hos_freq_cipher_dict1, hos_freq_cipher_dict_noT = numToFreqency(hos_freq_cipher_dict)
                hos_freq_plain_dict1, hos_freq_plain_dict_noT = numToFreqency(hos_freq_plain_dict)
                diag_freq_cipher_dict1, diag_freq_cipher_dict_noT = numToFreqency(diag_freq_cipher_dict)
                diag_freq_plain_dict1, diag_freq_plain_dict_noT = numToFreqency(diag_freq_plain_dict)

                # 先利用volume和freqency找到最独特的几个查询，利用不同时期的频率
                dis_dict = {}
                alpha = 0.03
                for key1 in set(hospital_cipher):
                    if key1 in hos_volume_cipher_dict and key1 in hos_freq_cipher_dict1:
                        volume1 = hos_volume_cipher_dict[key1]
                        freq1 = hos_freq_cipher_dict1[key1]
                        # print(freq1)
                        min_score = 100
                        for key2 in set(hospital_cipher):
                            if key2 != key1 and key2 in hos_volume_cipher_dict and key2 in hos_freq_cipher_dict1:
                                volume2 = hos_volume_cipher_dict[key2]
                                freq2 = hos_freq_cipher_dict1[key2]
                                v = abs(volume1 - volume2) *alpha
                                f = 0
                                for i in range(len(freq1)):
                                    f1 = freq1[data[i]]
                                    f2 = freq2[data[i]]
                                    f += abs(f1-f2)
                                f = f/len(freq1) * (1-alpha)
                                score = v+f
                                if score < min_score:
                                    min_score = score
                        dis_dict[key1] = min_score
                for key1 in set(diagnosis_cipher):
                    if key1 in diag_volume_cipher_dict and key1 in diag_freq_cipher_dict1:
                        volume1 = diag_volume_cipher_dict[key1]
                        freq1 = diag_freq_cipher_dict1[key1]
                        min_score = 100
                        for key2 in set(diagnosis_cipher):
                            if key2 != key1 and key2 in diag_volume_cipher_dict and key2 in diag_freq_cipher_dict1:
                                volume2 = diag_volume_cipher_dict[key2]
                                freq2 = diag_freq_cipher_dict1[key2]
                                v = abs(volume1 - volume2) *alpha
                                f = 0
                                for i in range(len(freq1)):
                                    f1 = freq1[data[i]]
                                    f2 = freq2[data[i]]
                                    f += abs(f1-f2)
                                f = f/len(freq1) * (1-alpha)
                                score = v+f
                                if score < min_score:
                                    min_score = score
                        dis_dict[key1] = min_score

                sorted_dis_dict = dict(sorted(dis_dict.items(), key=lambda item: item[1], reverse=True))
                print(len(set(hospital_cipher)))
                print(len(set(diagnosis_cipher)))
                print(len(sorted_dis_dict))
                # 对上述查询，找关键字
                pred_dict = {}
                count = 0
                for key1, value in sorted_dis_dict.items():
                    # count += 1
                    # if count == 500:
                    #     break
                    if key1 in hos_volume_cipher_dict and key1 in hos_freq_cipher_dict_noT:
                        volume1 = hos_volume_cipher_dict[key1]
                        freq1 = hos_freq_cipher_dict1[key1]
                        min_score = 100
                        candidate = key1
                        for key2 in set(hospital_plain):
                            if key2 in hos_volume_plain_dict and key2 in hos_freq_plain_dict1:
                                volume2 = hos_volume_plain_dict[key2]
                                freq2 = hos_freq_plain_dict1[key2]
                                v = abs(volume1 - volume2) *alpha
                                f = 0
                                for i in range(len(freq1)):
                                    f1 = freq1[data[i]]
                                    f2 = freq2[data[i]]
                                    f += abs(f1-f2)
                                f = f/len(freq1) * (1-alpha)
                                score = v+f
                                if score < min_score:
                                    min_score = score
                                    candidate = key2
                        pred_dict[key1] = candidate
                    else:
                        volume1 = diag_volume_cipher_dict[key1]
                        freq1 = diag_freq_cipher_dict1[key1]
                        min_score = 100
                        candidate = key1
                        for key2 in set(diagnosis_plain):
                            if key2 in diag_volume_plain_dict and key2 in diag_freq_plain_dict1:
                                volume2 = diag_volume_plain_dict[key2]
                                freq2 = diag_freq_plain_dict1[key2]
                                v = abs(volume1 - volume2) *alpha
                                f = 0
                                for i in range(len(freq1)):
                                    f1 = freq1[data[i]]
                                    f2 = freq2[data[i]]
                                    f += abs(f1-f2)
                                f = f/len(freq1) * (1-alpha)
                                score = v+f
                                if score < min_score:
                                    min_score = score
                                    candidate = key2
                        pred_dict[key1] = candidate
                for key, value in pred_dict.items():
                    if key == value:
                        value_mapping_ods[key] = value
                matrix_recovered = replace_values_with_none(matrix_cipher, value_mapping_ods)

                record_recovered = find_unique_rows_withNone(matrix_recovered)
                print(f"经过OPE+DET+SSE一共找到了{len(record_recovered)}条数据")

                re_c += record_count
                re_s_c += len(record_recovered)
                kw_c += keyword_count
                kw_s_c += len(value_mapping_ods)
                times += 1
                # if times == 5:
                #     break

            kw_c_aver = kw_c // times
            kw_s_c_aver = kw_s_c // times
            re_c_aver = re_c // times
            re_s_c_aver = re_s_c // times

            file_extension = os.path.splitext(file_name)[1]
            new_file_name = file_name.replace(file_extension, ".txt")
            outputPath = os.path.join(out,new_file_name)
            print(outputPath)
            with open(outputPath,"w") as f:
                f.write("record number: " + str(re_c_aver) + " successfully recovered number: " + str(re_s_c_aver) + " keywords number: " + str(kw_c_aver) + " successfully recovered keyword number: " + str(kw_s_c_aver) + "percent of file" + str(frac))

import sys # 引用上级目录中的文件
sys.path.append("/CCS2026/") 
import functions
import random
import time
import os
from collections import Counter


def CumulativeAttack(matrix_cipher, matrix_plain):
    startTime = time.time()
    # matrix_cipher = functions.read_csv_to_matrix(filePath)

    # 统计到底有多少个关键字
    selected_columns_ope_withoutid = ['Age', 'Admission Type', 'Length of stay', 'Risk']
    matrix_ope_withoutid = functions.generate_submatrix(matrix_cipher, selected_columns_ope_withoutid)
    keyword_count = functions.count_keywords(matrix_ope_withoutid[1:])

    # 生成密文矩阵
    selected_columns_ope = ['Record ID', 'Age', 'Admission Type', 'Length of stay', 'Risk']
    matrix_cipher_ope = functions.generate_submatrix(matrix_cipher, selected_columns_ope)

    # 生成明文矩阵
    # matrix_plain = functions.read_csv_to_matrix(filePathPlain)
    matrix_plain_ope = functions.generate_submatrix(matrix_plain, selected_columns_ope)

    value_mapping = {}
    # Age
    column_cipher_age = functions.extract_columns(matrix_cipher_ope, 'Age')
    age_cipher_count = Counter(column_cipher_age)
    keyword_count_cipher_age = len(age_cipher_count)

    column_plain_age = functions.extract_columns(matrix_plain_ope, 'Age')
    age_plain_count = Counter(column_plain_age)
    keyword_count_plain_age = len(age_plain_count)

    # if keyword_count_cipher_age == keyword_count_plain_age:
    #     sorted_cipher_age = sorted(set(column_cipher_age))
    #     sorted_plain_age = sorted(set(column_plain_age))
    #     mapping_ope_age = {key: value for key, value in zip(sorted_cipher_age, sorted_plain_age)}
    #     value_mapping.update(mapping_ope_age)
    # else:
    mapping_ope_age = functions.find_optimal_mapping_A4(column_cipher_age, column_plain_age)
    value_mapping.update(mapping_ope_age)

    # Admission Type
    column_cipher_admi = functions.extract_columns(matrix_cipher_ope, 'Admission Type')
    admi_cipher_count = Counter(column_cipher_admi)
    keyword_count_cipher_admi = len(admi_cipher_count)

    column_plain_admi = functions.extract_columns(matrix_plain_ope, 'Admission Type')
    admi_plain_count = Counter(column_plain_admi)
    keyword_count_plain_admi = len(admi_plain_count)

    # if keyword_count_cipher_admi == keyword_count_plain_admi:
    #     sorted_cipher_admi = sorted(set(column_cipher_admi))
    #     sorted_plain_admi = sorted(set(column_plain_admi))
    #     mapping_ope_admi = {key: value for key, value in zip(sorted_cipher_admi, sorted_plain_admi)}
    #     value_mapping.update(mapping_ope_admi)
    # else:
    mapping_ope_admi = functions.find_optimal_mapping_A4(column_cipher_admi, column_plain_admi)
    value_mapping.update(mapping_ope_admi)

    # Risk
    column_cipher_risk = functions.extract_columns(matrix_cipher_ope, 'Risk')
    risk_cipher_count = Counter(column_cipher_risk)
    keyword_count_cipher_risk = len(risk_cipher_count)

    column_plain_risk = functions.extract_columns(matrix_plain_ope, 'Risk')
    risk_plain_count = Counter(column_plain_risk)
    keyword_count_plain_risk = len(risk_plain_count)

    # if keyword_count_cipher_risk == keyword_count_plain_risk:
    #     sorted_cipher_risk = sorted(set(column_cipher_risk))
    #     sorted_plain_risk = sorted(set(column_plain_risk))
    #     mapping_ope_risk = {key: value for key, value in zip(sorted_cipher_risk, sorted_plain_risk)}
    #     value_mapping.update(mapping_ope_risk)
    # else:
    mapping_ope_risk = functions.find_optimal_mapping_A4(column_cipher_risk, column_plain_risk)
    value_mapping.update(mapping_ope_risk)

    # Length of Stay
    column_cipher_stay = functions.extract_columns(matrix_cipher_ope, 'Length of stay')
    stay_cipher_count = Counter(column_cipher_stay)
    keyword_count_cipher_stay = len(stay_cipher_count)

    column_plain_stay = functions.extract_columns(matrix_plain_ope, 'Length of stay')
    stay_plain_count = Counter(column_plain_stay)
    keyword_count_plain_stay = len(stay_plain_count)

    # if keyword_count_cipher_stay == keyword_count_plain_stay:
    #     sorted_cipher_stay = sorted(set(column_cipher_stay))
    #     sorted_plain_stay = sorted(set(column_plain_stay))
    #     mapping_ope_stay = {key: value for key, value in zip(sorted_cipher_stay, sorted_plain_stay)}
    #     value_mapping.update(mapping_ope_stay)
    # else:
    sorted_cipher_stay = sorted(column_cipher_stay, key=functions.custom_sort)
    sorted_plain_stay = sorted(column_plain_stay, key=functions.custom_sort)
    mapping_ope_stay = functions.find_optimal_mapping_A4(sorted_cipher_stay, sorted_plain_stay)
    value_mapping.update(mapping_ope_stay)


    endTime = time.time()
    totalTime = round(endTime - startTime, 2)

    accuracy = 0
    count = 0
    for key, value in value_mapping.items():
        if key == value:
            count += 1

    accuracy = count / keyword_count
    return totalTime, accuracy

if __name__ == '__main__':
    # filePath = "test.csv"
    # filePathPlain = "train.csv"
    # totalTime, accuracy = CumulativeAttack(filePathPlain, filePath)
    # print(accuracy)
    # root = "F:/Desktop/Supplementary experiments/New Dataset/"
    root = "dataset/text_508029.csv"
    filePathPlain = "dataset/2015.csv"
    # quarters = ['1q2010', '2q2010', '3q2010', '4q2010']
    # quarters = ['4q2010']
    out = 'result/single/output of CumulativeAttack-NKW15.txt'
    base = [500, 725, 1050, 1525, 2210, 3205, 4645, 6735, 9765, 14160, 20530, 29770, 43170, 62600, 90750, 131600, 190850, 276750, 401300, 508029]
    matrix = functions.read_csv_to_matrix(root)
    matrix_plain = functions.read_csv_to_matrix(filePathPlain)
    # 打开输出文件
    with open(out, 'w', encoding='utf-8') as f:
        for i in base:
            print(i)
            selected_columns_ope_withoutid = ['Age', 'Admission Type', 'Length of stay', 'Risk']
            total_times = []
            accuracies = []
            keyword_counts = []
            # 每个文件运行10次取平均值
            for _ in range(50):
                matrix_cipher = functions.random_extract(matrix, i)
                totalTime, accuracy = CumulativeAttack(matrix_cipher, matrix_plain)
                total_times.append(totalTime)
                accuracies.append(accuracy)
                # keyword_counts.append(keyword_count)

            # 计算平均值
            avg_time = sum(total_times) / 50
            avg_accuracy = sum(accuracies) / 50
            # avg_count = sum(keyword_counts) / 50

            # 将结果写入文件
            f.write(f"文件路径: 4q2010\\text {i}.txt, 执行时间: {avg_time}秒, 准确率: {avg_accuracy}\n")
            f.write("-" * 50 + "\n")

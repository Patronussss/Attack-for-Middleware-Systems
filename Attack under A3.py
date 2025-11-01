import sys
sys.path.append("/media/ices/machenrry/zl/Attack for DataBlinder/") 
import functions
from ours.EnhancedCumulativeAttack import EnhancedCumulativeAttack
from ours.AttackusingAuxiliary import AttackUsingAuxiliaryWeight
from ours.EnhancedFrequencyAnalysisAttack import EnhancedFrequencyAnalysisAttack
import pandas as pd
import numpy as np
from collections import Counter
import time
import os
import networkx as nx
from tqdm import tqdm

def process_matrix_from_mapping(matrix, columns, replacement_dict):
    header = matrix[0]
    column_indices = [header.index(col) for col in columns]

    for row in matrix[1:]:
        for index in column_indices:
            value = row[index]
            if value in replacement_dict and replacement_dict[value] == value:
                continue
            else:
                row[index] = 'zlzlzl'
    return matrix

# index_list = [4]

# 数据集为PUDF2010q4和2015时的实验
filePathPlain = "dataset/2015.csv"
matrix_plain = functions.read_csv_to_matrix(filePathPlain)
out = 'result/A4/ours/'
# for il in index_list:
root = "dataset/text_508029.csv"
# base = [725, 1050, 1525, 2210, 3205, 4645, 6735, 9765, 14160, 20530, 29770, 43170, 62600, 90750, 131600, 190850, 276750, 401300, 508029] 
base = [500, 725, 1050, 1525, 2210, 3205, 4645, 6735, 9765, 14160, 20530, 29770, 43170, 62600, 90750, 131600, 190850, 276750, 401300, 508029] 
# year = '2017'
# quarter = '4'

# # # 数据集为PUDF2019q4和2018时的实验
# filePathPlain = f"E:/code/Searchable Encryption/dataset/PUDF/ZLDS/{year}/{year}_withoutQ{quarter}.csv"
# matrix_plain = functions.read_csv_to_matrix(filePathPlain)
# out = f'F:/Desktop/Supplementary experiments/result/20250924/A4/4q{year} output of A4_aux_{year}_withoutQ{quarter}/'
# # for il in index_list:
# root = f"E:/code/Searchable Encryption/dataset/PUDF/ZLDS/{year}/PUDF_base1_{quarter}q{year}.csv"
# base = [500, 36696, 72892, 109088, 145284, 181480, 217676, 253872, 290068, 326264, 362460, 398656, 434852, 471048, 507244, 543440, 579636, 615832, 652028, 688227] 
# base = [500, 732, 1071, 1567, 2293, 3355, 4910, 7183, 10510, 15376, 22495, 32910, 48147, 70439, 103052, 150765, 220569, 322703, 472115, 688227] # 取对数版本

matrix = functions.read_csv_to_matrix(root)
# base = 'F:\\Desktop\\Supplementary experiments\\New Dataset\\' + str(il)+ 'q2010/'

if not os.path.exists(out):
    os.makedirs(out)
for i in base:
    print(i)
    recovered_keywords = []
    keywords_count = []
    for _ in range(50):
        matrix_cipher = functions.random_extract(matrix, i)
        value_mapping = {}
        selected_columns_ope_withoutid = ['Age', 'Admission Type', 'Length of stay', 'Risk']
        mapping_ope, time_ope, accuracy_ope, keyword_count_ope = EnhancedCumulativeAttack(matrix_cipher, matrix_plain, selected_columns_ope_withoutid, [0.5, 0.1, 0.4])
        value_mapping.update(mapping_ope)
        # DET的部分，计算频率和种类
        selected_columns_det = ['Gender', 'Race']
        mapping_det, time_det, accuracy_det, keyword_count_det, count_det = EnhancedFrequencyAnalysisAttack(matrix_cipher, matrix_plain, selected_columns_det,"PUDF", [1,1])
        value_mapping.update(mapping_det)
        
        # SSE
        selected_columns_sse = ['Hospital','Pincipal Diagnosis']
        specific_columns = ["Age", "Gender",  "Risk", "Admission Type", "Race"]
        matrix_cipher_sse = functions.generate_submatrix(matrix_cipher, selected_columns_sse)
        matrix_cipher_sse_after_mapping = process_matrix_from_mapping(matrix_cipher, specific_columns, value_mapping)

        mapping, totalTime, accuracy, keyword_count_sse = AttackUsingAuxiliaryWeight(matrix_cipher_sse_after_mapping, matrix_plain, selected_columns_sse, "PUDF", [35, 45, 20])

        value_mapping.update(mapping)

        count = 0
        for key, value in value_mapping.items():
            if key == value:
                count += 1
        keyword_number = keyword_count_ope + keyword_count_det + keyword_count_sse
        keywords_count.append(keyword_number)
        recovered_keywords.append(count)
    
    avg_keyword_number = sum(keywords_count) / 50
    avg_recovered_keywords = sum(recovered_keywords) / 50

    new_file_name = "text_" + str(i) + ".txt"
    outputPath = os.path.join(out,new_file_name)
    print(outputPath)
    with open(outputPath,"w") as f:
        f.write("keywords number: " + str(avg_recovered_keywords) + " successfully recovered keyword number: " + str(avg_keyword_number))




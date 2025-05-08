import sys
sys.path.append("F:/Desktop/Supplementary experiments") 
import functions
from ours.EnhancedCumulativeAttack import EnhancedCumulativeAttack
from ours.AttackusingAuxiliary import AttackUsingAuxiliary
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

index_list = [4]
filePathPlain = "F:/Desktop/Attack for Datablinder/2015/2015.csv"
matrix_plain = functions.read_csv_to_matrix(filePathPlain)

for il in index_list:
    base = 'F:\\Desktop\\Supplementary experiments\\New Dataset\\' + str(il)+ 'q2010/'
    
    file_list = os.listdir(base)
    
    for file_name in sorted(file_list):
        # if file_name[:-4] not in lists:
        for time in range(1,6):
            out = 'F:/Desktop/Supplementary experiments/result/A4/' + str(il) + 'q2010 output of A4/'+ str(time) + '/'
            if not os.path.exists(out):
                os.makedirs(out)
            lists = [i[:-4] for i in os.listdir(out)]
            filePath = os.path.join(base,file_name)
            matrix_cipher = functions.read_csv_to_matrix(filePath)
            
            value_mapping = {}
            selected_columns_ope_withoutid = ['Age', 'Admission Type', 'Length of stay', 'Risk']
            mapping_ope, time_ope, accuracy_ope, keyword_count_ope = EnhancedCumulativeAttack(filePath, matrix_plain, selected_columns_ope_withoutid )
            value_mapping.update(mapping_ope)
            # DET的部分，计算频率和种类
            selected_columns_det = ['Gender', 'Race']
            matrix_cipher_det = functions.generate_submatrix(matrix_cipher, selected_columns_det)
            matrix_plain_det = functions.generate_submatrix(matrix_plain, selected_columns_det)
            # 计算频率
            element_cipher_det = functions.column_frequencies(matrix_cipher_det[1:])
            element_plain_det = functions.column_frequencies(matrix_plain_det[1:])

            for key, dict1 in element_cipher_det.items():
                dict2 = element_plain_det[key]
                temp = functions.find_closest_mapping(dict1, dict2)
                value_mapping.update(temp)
        
            
            # SSE
            selected_columns_sse = ['Hospital','Pincipal Diagnosis']
            specific_columns = ["Age", "Gender",  "Risk", "Admission Type", "Race"]
            matrix_cipher_sse = functions.generate_submatrix(matrix_cipher, selected_columns_sse)
            matrix_cipher_sse_after_mapping = process_matrix_from_mapping(matrix_cipher, specific_columns, value_mapping)

            mapping, totalTime, accuracy = AttackUsingAuxiliary(matrix_cipher_sse_after_mapping, matrix_plain, selected_columns_sse, "PUDF")

            value_mapping.update(mapping)

            count = 0
            for key, value in value_mapping.items():
                if key == value:
                    count += 1
            keyword_number = keyword_count_ope + functions.count_keywords(matrix_cipher_det[1:]) + functions.count_keywords(matrix_cipher_sse[1:])

            file_extension = os.path.splitext(file_name)[1]
            new_file_name = file_name.replace(file_extension, ".txt")
            outputPath = os.path.join(out,new_file_name)
            print(outputPath)
            with open(outputPath,"w") as f:
                f.write("keywords number: " + str(keyword_number) + " successfully recovered keyword number: " + str(count))




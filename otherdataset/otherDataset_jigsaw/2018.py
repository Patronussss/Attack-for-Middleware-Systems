import sys
sys.path.append("/media/ices/machenrry/zl/Attack for DataBlinder/") 
import functions
from ours.EnhancedCumulativeAttack import EnhancedCumulativeAttack
from ours.AttackusingAuxiliary import AttackUsingAuxiliaryWeight
from ours.EnhancedFrequencyAnalysisAttack import EnhancedFrequencyAnalysisAttack
from NKW15andSSESchemes.AttributeRecoverAttack import AttributeRecoverAttack
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

filePathPlain = "dataset/2015.csv"
matrix_plain = functions.read_csv_to_matrix(filePathPlain)
out = 'result/other dataset/2018Q4/'
root = "dataset/PUDF_base1_4q2018.csv"
base = [500, 11816, 23132, 34448, 45764, 57080, 68396, 79712, 91028, 102344, 113657]

matrix = functions.read_csv_to_matrix(root)

if not os.path.exists(out):
    os.makedirs(out)

selected_columns_plain_for_attr_recovery = ['Age', 'Admission Type', 'Length of stay', 'Risk', 'Discharge', 'Gender', 'Race']

for i in base:
    print(i)
    recovered_keywords = []
    keywords_count = []
    for _ in range(50):
        matrix_cipher = functions.random_extract(matrix, i)
        
        cipher_columns_all = matrix_cipher[0][1:]
        
        matrix_cipher_for_attr_recovery = functions.generate_submatrix(matrix_cipher, ['Record ID'] + cipher_columns_all)
        matrix_plain_for_attr_recovery = functions.generate_submatrix(matrix_plain, selected_columns_plain_for_attr_recovery)
        
        attr_mapping, _, _ = AttributeRecoverAttack(matrix_cipher_for_attr_recovery, matrix_plain_for_attr_recovery, ['Record ID'] + cipher_columns_all, selected_columns_plain_for_attr_recovery)
        
        ope_target_cols = ['Age', 'Admission Type', 'Length of stay', 'Risk']
        det_target_cols = ['Gender', 'Race']
        sse_target_cols = ['Hospital','Pincipal Diagnosis']
        
        selected_columns_ope_withoutid = [cipher_col for cipher_col, plain_col in attr_mapping.items() if plain_col in ope_target_cols]
        selected_columns_det = [cipher_col for cipher_col, plain_col in attr_mapping.items() if plain_col in det_target_cols]
        selected_columns_sse = [cipher_col for cipher_col, plain_col in attr_mapping.items() if plain_col in sse_target_cols]
        
        selected_columns_ope = ['Record ID'] + selected_columns_ope_withoutid
        
        value_mapping = {}
        mapping_ope, time_ope, accuracy_ope, keyword_count_ope = EnhancedCumulativeAttack(matrix_cipher, matrix_plain, selected_columns_ope_withoutid, [0.5, 0.1, 0.4])
        value_mapping.update(mapping_ope)
        
        mapping_det, time_det, accuracy_det, keyword_count_det, count_det = EnhancedFrequencyAnalysisAttack(matrix_cipher, matrix_plain, selected_columns_det,"PUDF", [1,1])
        value_mapping.update(mapping_det)
        
        specific_columns = [cipher_col for cipher_col, plain_col in attr_mapping.items() if plain_col in ["Age", "Gender", "Risk", "Admission Type", "Race"]]
        matrix_cipher_sse = functions.generate_submatrix(matrix_cipher, selected_columns_sse)
        matrix_cipher_sse_after_mapping = process_matrix_from_mapping(matrix_cipher, specific_columns, value_mapping)

        mapping, totalTime, accuracy, keyword_count_sse = AttackUsingAuxiliaryWeight(matrix_cipher_sse_after_mapping, matrix_plain, selected_columns_sse, "PUDF", [35,45,20])

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
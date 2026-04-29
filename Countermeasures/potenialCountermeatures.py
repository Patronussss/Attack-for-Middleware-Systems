import sys
sys.path.append("/CCS2026/") 
import functions
from ours.EnhancedCumulativeAttack import EnhancedCumulativeAttack
from Countermeasures.PotenialAttackUsingAuxiliary import AttackUsingAuxiliary
from Countermeasures.PotenialAttackUsingAuxiliary import EnhancedFrequencyAnalysisAttack
from ours.AttributeRecoveryAttack import AttributeRecoveryAttack
import pandas as pd
import numpy as np
from collections import Counter
import time
import os
import networkx as nx
from tqdm import tqdm

def process_matrix_from_mapping(matrix, columns):
    header = matrix[0]
    column_indices = [header.index(col) for col in columns]

    for row in matrix[1:]:
        for index in column_indices:
            row[index] = 'zlzlzl'
    
    return matrix

index_list = [4]
filePathPlain = f"/media/ices/machenrry/zl/Attack for DataBlinder/dataset/2015.csv"
matrix_plain = functions.read_csv_to_matrix(filePathPlain)

out = 'result/CCS/A2_Potenial/'
root = "dataset/text_508029.csv"
base = [725, 1050, 1525, 2210, 3205, 4645, 6735, 9765, 14160, 20530, 29770, 43170, 62600, 90750, 131600, 190850, 276750, 401300, 508029]
matrix = functions.read_csv_to_matrix(root)

for i in base:
    print(i)
    recovered_keywords = []
    keywords_count = []
    for _ in range(5):
        matrix_cipher = functions.random_extract(matrix, i)
        # num_rows_to_extract = int((len(matrix_cipher)-1) * 0.9)
        # matrix_plain = functions.random_submatrix(matrix_cipher, num_rows_to_extract)
        value_mapping = {}

        selected_columns_cipher_for_attr_recovery = ['Age', 'Admission Type', 'Length of stay', 'Risk', 'Discharge', 'Gender', 'Race', 'Hospital', 'Pincipal Diagnosis']
        selected_columns_plain_for_attr_recovery = ['Age', 'Admission Type', 'Length of stay', 'Risk', 'Discharge', 'Gender', 'Race']
        
        matrix_cipher_for_attr_recovery = functions.generate_submatrix(matrix_cipher, selected_columns_cipher_for_attr_recovery)
        matrix_plain_for_attr_recovery = functions.generate_submatrix(matrix_plain, selected_columns_plain_for_attr_recovery)
        
        attr_mapping, _, _, _ = AttributeRecoveryAttack(
            matrix_cipher_for_attr_recovery, 
            matrix_plain_for_attr_recovery, 
            selected_columns_cipher_for_attr_recovery, 
            selected_columns_plain_for_attr_recovery
        )

        selected_columns_ope_withoutid = []
        ope_target_cols = ['Age', 'Admission Type', 'Length of stay', 'Risk']
        for cipher_col, plain_col in attr_mapping.items():
            if plain_col in ope_target_cols:
                selected_columns_ope_withoutid.append(cipher_col)

        selected_columns_det = []
        det_target_cols = ['Gender', 'Race']
        for cipher_col, plain_col in attr_mapping.items():
            if plain_col in det_target_cols:
                selected_columns_det.append(cipher_col)

        selected_columns_sse = []
        sse_target_cols = ['Hospital', 'Pincipal Diagnosis']
        for cipher_col, plain_col in attr_mapping.items():
            if plain_col in sse_target_cols:
                selected_columns_sse.append(cipher_col)

        specific_columns = []
        specific_target_cols = ["Age", "Gender", "Risk", "Admission Type", "Race"]
        for cipher_col, plain_col in attr_mapping.items():
            if plain_col in specific_target_cols:
                specific_columns.append(cipher_col)

        mapping_ope, time_ope, accuracy_ope, keyword_count_ope = EnhancedCumulativeAttack(matrix_cipher, matrix_plain, selected_columns_ope_withoutid, [0.5, 0.5, 0])
        value_mapping.update(mapping_ope)

        mapping_det, time_det, accuracy_det, keyword_count_det, count_det = EnhancedFrequencyAnalysisAttack(matrix_cipher, matrix_plain, selected_columns_det, "PUDF")
        value_mapping.update(mapping_det)

        matrix_cipher_sse = functions.generate_submatrix(matrix_cipher, selected_columns_sse)
        matrix_cipher_sse_after_mapping = process_matrix_from_mapping(matrix_cipher, specific_columns)

        mapping, totalTime, accuracy = AttackUsingAuxiliary(matrix_cipher_sse_after_mapping, matrix_plain, selected_columns_sse, "PUDF")

        value_mapping.update(mapping)

        count = 0
        for key, value in value_mapping.items():
            if key == value:
                count += 1
        keyword_number = keyword_count_ope + keyword_count_det + functions.count_keywords(matrix_cipher_sse[1:])
        keywords_count.append(keyword_number)
        recovered_keywords.append(count)
    
    avg_keyword_number = sum(keywords_count) / 5
    avg_recovered_keywords = sum(recovered_keywords) / 5

    new_file_name = "text_" + str(i) + ".txt"
    outputPath = os.path.join(out, new_file_name)
    print(outputPath)
    with open(outputPath, "w") as f:
        f.write("keywords number: " + str(avg_recovered_keywords) + " successfully recovered keyword number: " + str(avg_keyword_number))

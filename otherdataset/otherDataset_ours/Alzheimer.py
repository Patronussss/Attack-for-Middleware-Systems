import sys
sys.path.append("/media/data1/mcr/zl/zl/AttackForDataBlinder/") 
import functions
from ours.EnhancedCumulativeAttack import EnhancedCumulativeAttack
from ours.AttackusingAuxiliary import AttackUsingAuxiliaryWeight
from ours.EnhancedFrequencyAnalysisAttack import EnhancedFrequencyAnalysisAttack
from ours.AttributeRecoveryAttack import AttributeRecoveryAttack
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

index_list = [5]
for il in index_list:
    
    if il == 4:
        out = 'result/other dataset/2010Q4/'
        filePathPlain = "dataset/2015.csv"
        root = "dataset/text_508029.csv"
    elif il == 5:
        out = 'result/CCS/A4_Alzheimer/'
        filePathPlain = "dataset/Alzheimer_plain.csv"
        root = "dataset/Alzheimer_cipher.csv"
    # for il in index_list:

    # 差几个数据集
    

    base = [500, 11816, 23132, 34448, 45764, 57080, 68396, 79712, 91028, 102344, 113657]
    matrix_plain = functions.read_csv_to_matrix(filePathPlain)
    
    matrix = functions.read_csv_to_matrix(root)

    if not os.path.exists(out):
        os.makedirs(out)
    for ind in base:
        print(ind)
        recovered_keywords = []
        keywords_count = []
        for _ in range(5):
            matrix_cipher = functions.random_extract(matrix, ind)
            value_mapping = {}
            
            if il != 5:
                selected_columns_plain_for_attr_recovery = ['Age', 'Admission Type', 'Length of stay', 'Risk', 'Discharge', 'Gender', 'Race']
            else:
                selected_columns_plain_for_attr_recovery = ['YearStart', 'LocationAbbr', 'Stratification2', 'Low_Confidence_Limit', 'High_Confidence_Limit', 'Class', 'DataValueTypeID']
            
            cipher_columns_all = matrix_cipher[0][1:]
            
            matrix_cipher_for_attr_recovery = functions.generate_submatrix(matrix_cipher, ['Record ID'] + cipher_columns_all)
            matrix_plain_for_attr_recovery = functions.generate_submatrix(matrix_plain, selected_columns_plain_for_attr_recovery)
            
            attr_mapping, _, _, _ = AttributeRecoveryAttack(
                matrix_cipher_for_attr_recovery, 
                matrix_plain_for_attr_recovery, 
                ['Record ID'] + cipher_columns_all, 
                selected_columns_plain_for_attr_recovery
            )
            
            selected_columns_ope_withoutid = []
            if il != 5:
                ope_target_cols = ['Age', 'Admission Type', 'Length of stay', 'Risk']
            else:
                ope_target_cols = ['YearStart', 'LocationAbbr', 'Stratification2', 'Low_Confidence_Limit', 'High_Confidence_Limit']
            for cipher_col, plain_col in attr_mapping.items():
                if plain_col in ope_target_cols:
                    selected_columns_ope_withoutid.append(cipher_col)
            
            mapping_ope, time_ope, accuracy_ope, keyword_count_ope = EnhancedCumulativeAttack(matrix_cipher, matrix_plain, selected_columns_ope_withoutid, [0.5,0.1,0.4])
            value_mapping.update(mapping_ope)
            
            selected_columns_det = []
            if il != 5:
                det_target_cols = ['Gender', 'Race']
            else:
                det_target_cols = ['Class', 'DataValueTypeID']
            for cipher_col, plain_col in attr_mapping.items():
                if plain_col in det_target_cols:
                    selected_columns_det.append(cipher_col)
            
            if il != 5:
                mapping_det, time_det, accuracy_det, keyword_count_det, count_det = EnhancedFrequencyAnalysisAttack(matrix_cipher, matrix_plain, selected_columns_det,"PUDF",[1,1])
            else:
                mapping_det, time_det, accuracy_det, keyword_count_det, count_det = EnhancedFrequencyAnalysisAttack(matrix_cipher, matrix_plain, selected_columns_det,"Alzheimer",[1,1])
            value_mapping.update(mapping_det)
            
            selected_columns_sse = []
            if il != 5:
                sse_target_cols = ['Hospital', 'Pincipal Diagnosis']
            else:
                sse_target_cols = ["Topic"]
            for cipher_col, plain_col in attr_mapping.items():
                if plain_col in sse_target_cols:
                    selected_columns_sse.append(cipher_col)
            
            specific_columns = []
            if il != 5:
                specific_target_cols = ["Age", "Gender", "Risk", "Admission Type", "Race"]
            else:
                specific_target_cols = ['YearStart', 'LocationAbbr', 'Stratification2', 'Class', 'DataValueTypeID']
            for cipher_col, plain_col in attr_mapping.items():
                if plain_col in specific_target_cols:
                    specific_columns.append(cipher_col)
            
            matrix_cipher_sse = functions.generate_submatrix(matrix_cipher, selected_columns_sse)
            matrix_cipher_sse_after_mapping = process_matrix_from_mapping(matrix_cipher, specific_columns, value_mapping)
            if il != 5:
                mapping, totalTime, accuracy, keyword_count_sse = AttackUsingAuxiliaryWeight(matrix_cipher_sse_after_mapping, matrix_plain, selected_columns_sse, "PUDF", [0.35, 0.45, 0.2])
            if il == 5:
                mapping, totalTime, accuracy, keyword_count_sse = AttackUsingAuxiliaryWeight(matrix_cipher_sse_after_mapping, matrix_plain, selected_columns_sse, "Alzheimer", [5, 60, 35])
            value_mapping.update(mapping)

            count = 0
            for key, value in value_mapping.items():
                if key == value:
                    count += 1
            keyword_number = keyword_count_ope + keyword_count_det + keyword_count_sse
            keywords_count.append(keyword_number)
            recovered_keywords.append(count)
        
        avg_keyword_number = sum(keywords_count) / 5
        avg_recovered_keywords = sum(recovered_keywords) / 5

        new_file_name = "text_" + str(ind) + ".txt"
        outputPath = os.path.join(out,new_file_name)
        print(outputPath)
        with open(outputPath,"w") as f:
            f.write("keywords number: " + str(avg_recovered_keywords) + " successfully recovered keyword number: " + str(avg_keyword_number))
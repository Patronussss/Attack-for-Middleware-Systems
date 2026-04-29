import sys
sys.path.append("/media/data1/mcr/zl/zl/AttackForDataBlinder") 
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

root = "/media/data1/mcr/zl/zl/AttackForDataBlinder/dataset/text_508029.csv"

base = [5000, 10000, 100000, 225000, 337500, 508029]
frac_list = [0.7, 0.9]
matrix = functions.read_csv_to_matrix(root)

for ind in base:
    for frac in frac_list:
        for time_iter in range(50):
            out = f'/media/data1/mcr/zl/zl/AttackForDataBlinder/result/TIFS/A1-ours/' + str(frac) + '/' + str(time_iter) + '/'
            if not os.path.exists(out):
                os.makedirs(out)
            matrix_c = functions.random_extract(matrix, ind)
            
            selected_columns_cipher = ['Record ID', 'Age', 'Admission Type', 'Length of stay', 
            'Risk', 'Gender', 'Race','Hospital', 'Pincipal Diagnosis']
            matrix_cipher = functions.generate_submatrix(matrix_c, selected_columns_cipher)
            num_rows_to_extract =  int((len(matrix_cipher)-1) * frac)
            matrix_plain = functions.random_submatrix(matrix_cipher, num_rows_to_extract)
            functions.replace_nested_inplace(matrix_plain, 'American_Indian/Eskimo/Aleut', 'American Indian/Eskimo/Aleut')
            functions.replace_nested_inplace(matrix_plain, 'Asian_or_Pacific_Islander', 'Asian or Pacific Islander')

            record_count = len(matrix_plain)-1
            keyword_count = functions.count_keywords(functions.generate_submatrix(matrix_cipher,
            ['Age', 'Admission Type', 'Length of stay', 'Risk', 
            'Gender', 'Race','Hospital', 'Pincipal Diagnosis'])[1:])

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

            print("Recovered attribute mapping:")
            for cipher_col, plain_col in attr_mapping.items():
                print(f"  {cipher_col} -> {plain_col}")

            selected_columns_ope_withoutid_cipher = []
            ope_target_cols = ['Age', 'Admission Type', 'Length of stay', 'Risk']
            for cipher_col, plain_col in attr_mapping.items():
                if plain_col in ope_target_cols:
                    selected_columns_ope_withoutid_cipher.append(cipher_col)

            selected_columns_det_cipher = []
            det_target_cols = ['Gender', 'Race']
            for cipher_col, plain_col in attr_mapping.items():
                if plain_col in det_target_cols:
                    selected_columns_det_cipher.append(cipher_col)

            selected_columns_sse_cipher = []
            sse_target_cols = ['Hospital', 'Pincipal Diagnosis']
            for cipher_col, plain_col in attr_mapping.items():
                if plain_col in sse_target_cols:
                    selected_columns_sse_cipher.append(cipher_col)

            specific_columns_cipher = []
            specific_target_cols = ["Age", "Gender", "Risk", "Admission Type", "Race"]
            for cipher_col, plain_col in attr_mapping.items():
                if plain_col in specific_target_cols:
                    specific_columns_cipher.append(cipher_col)

            mapping_ope, time_ope, accuracy_ope, keyword_count_ope = EnhancedCumulativeAttack(
                matrix_cipher, matrix_plain, selected_columns_ope_withoutid_cipher, [0.5, 0.1, 0.4]
            )
            value_mapping.update(mapping_ope)

            mapping_det, time_det, accuracy_det, keyword_count_det, count_det = EnhancedFrequencyAnalysisAttack(
                matrix_cipher, matrix_plain, selected_columns_det_cipher, "PUDF", [1,1]
            )
            value_mapping.update(mapping_det)

            matrix_cipher_sse = functions.generate_submatrix(matrix_cipher, selected_columns_sse_cipher)
            matrix_cipher_sse_after_mapping = process_matrix_from_mapping(matrix_cipher, specific_columns_cipher, value_mapping)

            mapping, totalTime, accuracy, keyword_count_sse = AttackUsingAuxiliaryWeight(
                matrix_cipher_sse_after_mapping, matrix_plain, selected_columns_sse_cipher, "PUDF", [35, 45, 20]
            )

            value_mapping.update(mapping)

            count = 0
            for key, value in value_mapping.items():
                if key == value:
                    count += 1
            keyword_number = keyword_count_ope + keyword_count_det + keyword_count_sse

            selected_columns_cipher_iods = ['Record ID'] + selected_columns_ope_withoutid_cipher + selected_columns_det_cipher + selected_columns_sse_cipher

            matrix_cipher_iods = functions.generate_submatrix(matrix_cipher, selected_columns_cipher_iods)
            matrix_recovered_sse = functions.replace_values_with_none(matrix_cipher_iods, value_mapping)

            record_recovered_iods = functions.find_unique_rows_withNone(matrix_recovered_sse)

            record_plain = [row[0] for row in matrix_plain]
            record = list(set(record_plain) & set(record_recovered_iods))

            print(f"成功恢复了{len(record)}条数据")
            print(f"一共拥有{len(matrix_plain)}条记录")
            print(f"恢复记录比例：{len(record) / len(matrix_plain)}")

            value_mapping_id = {}
            value_mapping_id.update(value_mapping)
            for i in record:
                value_mapping_id[i] = i

            re_c = record_count
            re_s_c = len(value_mapping_id) - len(value_mapping)
            kw_c = keyword_count
            kw_s_c = len(value_mapping)

            new_file_name = "text_" + str(ind) + ".txt"
            outputPath = os.path.join(out, new_file_name)
            print(outputPath)
            with open(outputPath, "w") as f:
                f.write("record number: " + str(re_c) + " successfully recovered number: " + str(re_s_c) + " keywords number: " + str(kw_c) + " successfully recovered keyword number: " + str(kw_s_c) + " percent of file: " + str(frac))

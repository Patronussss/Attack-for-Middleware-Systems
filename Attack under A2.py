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

index_list = [4]
frac_list = [0.1, 0.3, 0.5, 0.7, 0.9]

for frac in frac_list:
    for il in index_list:
        base = 'F:\\Desktop\\Supplementary experiments\\New Dataset\\'+ str(il)+ 'q2010/'
        # target = '/Users/cherry/Desktop/zengli/'+ str(il)+ 'q2010plain/'
        file_list = os.listdir(base)
        for file_name in sorted(file_list):
            for time in range(1,6):
                out = 'F:/Desktop/Supplementary experiments/result/A3/'+ str(il)+ 'q2010 output of A3/' + str(frac) + '/' + str(time) + '/'
                if not os.path.exists(out):
                    os.makedirs(out)
                filePath = os.path.join(base,file_name)
                matrix = functions.read_csv_to_matrix(filePath)
                selected_columns_cipher = ['Record ID', 'Age', 'Admission Type', 'Length of stay', 
                'Risk', 'Gender', 'Race','Hospital', 'Pincipal Diagnosis']
                matrix_cipher = functions.generate_submatrix(matrix, selected_columns_cipher)
                num_rows_to_extract =  int((len(matrix_cipher)-1) * frac)
                matrix_plain = functions.random_submatrix(matrix_cipher, num_rows_to_extract)

                record_count = len(matrix_plain)-1 #能够过恢复的最多的record 记录（剩下的没有对照的identifire就算恢复了所有数据也没法比较）
                keyword_count = functions.count_keywords(functions.generate_submatrix(matrix_cipher,
                ['Age', 'Admission Type', 'Length of stay', 'Risk', 
                'Gender', 'Race','Hospital', 'Pincipal Diagnosis'])[1:]) #能够恢复的关键字个数，由于只能拿到部分明文数据库，因此能够拿到的关键字个数和明文数据库拥有的明文个数相关

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

                selected_columns_od = ['Record ID', 'Age', 'Admission Type','Length of stay', 'Risk', 'Gender', 'Race']
                matrix_plain_od = functions.generate_submatrix(matrix_plain, selected_columns_od)
                matrix_cipher_od = functions.generate_submatrix(matrix_cipher, selected_columns_od)
                matrix_recovered_od = functions.replace_values_with_none(matrix_cipher_od, value_mapping)
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

                unique_value_cipher_od = functions.find_unique_value(list_cipher_od)
                unique_value_plain_od = functions.find_unique_value(list_plain_od)

                unique_rows = [v for v in unique_value_plain_od if v in unique_value_cipher_od]

                recovered_rows = {}
                for row in unique_rows:
                    id_cipher = record_cipher[list_cipher_od.index(row)]
                    id_plain = record_plain[list_plain_od.index(row)]
                    recovered_rows[id_cipher] = id_plain

                recordid_cipher = functions.create_rowid_dict(matrix_cipher, matrix_cipher[0], 'Record ID')
                recordid_plain = functions.create_rowid_dict(matrix_plain, matrix_plain[0], 'Record ID')
                for id_c, id_p in recovered_rows.items():
                    # index_cipher = recordid_cipher[id_c]
                    row_cipher = matrix_cipher[recordid_cipher[id_c]]
                    index_plain = recordid_plain[id_p]
                    row_plain = matrix_plain[index_plain]
                    for i in range(1, len(row_cipher)):
                        value_mapping[row_cipher[i]] = row_plain[i]

                matrix_recovered = functions.replace_values_with_none(matrix_cipher_od, value_mapping)
                record_recovered = functions.find_unique_rows_withNone(matrix_recovered)

                selected_column_sse = ['Record ID', 'Hospital', 'Pincipal Diagnosis']
                matrix_cipher_sse = functions.generate_submatrix(matrix_cipher, selected_column_sse)
                matrix_plain_sse = functions.generate_submatrix(matrix_plain, selected_column_sse)

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
                    decimal_sequence, keywords = functions.generate_decimal_sequence(matrix_recovered_sse)
                    # 创建一个映射，将十进制值映射回关键字
                    decimal_to_keyword = {decimal: keyword for decimal, keyword in zip(decimal_sequence, keywords)}
                    # 找出每个序列中独特的值（只出现一次的值）
                    unique_values = [decimal for decimal in decimal_sequence if decimal_sequence.count(decimal) == 1]
                    # 找出两个序列中相同的独特值
                    common_unique_values = set(unique_values) & set(unique_values)

                    # 记录这些相同的值对应的关键字
                    common_keywords = {decimal_to_keyword[value]: decimal_to_keyword[value] for value in common_unique_values if value in decimal_to_keyword and value in decimal_to_keyword}
                    value_mapping.update(common_keywords)
                    # print(len(value_mapping))

                    keyword_count_sse = functions.count_keywords(functions.generate_submatrix(matrix_plain,[
                        'Length of stay',
                        'Age', 'Admission Type', 'Risk',
                        'Gender', 'Race',
                        'Hospital', 'Pincipal Diagnosis'
                        ])[1:]) 
                    print(f"OPE+DET+SSE中一共包含{keyword_count_sse}个关键字")

                    # 返回到矩阵中看能不能获得更多
                    print(f"经过volume一共找到了{len(common_keywords)}个关键字")

                selected_columns_cipher_iods = ['Record ID', 'Age', 'Admission Type', 'Length of stay', 
                'Risk', 'Gender', 'Race', 'Hospital', 'Pincipal Diagnosis']

                matrix_cipher_iods = functions.generate_submatrix(matrix_cipher, selected_columns_cipher_iods)  # 经过OPE和DET后恢复了多少个记录
                matrix_recovered_sse = functions.replace_values_with_none(matrix_cipher_iods, value_mapping)

                record_recovered_iods = functions.find_unique_rows_withNone(matrix_recovered_sse)

                # 创建字典记录行号和对应的record id列的值
                record_plain = [row[0] for row in matrix_plain]
                # print(matrix_plain[0])
                # record_plain = create_rowid_dict(matrix_plain, matrix_plain[0], 'Record ID')
                record = list(set(record_plain) & set(record_recovered_iods))

                print(f"成功恢复了{len(record)}条数据")
                print(f"一共拥有{len(matrix_plain)}条记录")
                print(f"恢复记录比例：{len(record) / len(matrix_plain)}")

                value_mapping_id = {}
                value_mapping_id.update(value_mapping)
                for i in record:
                    value_mapping_id[i] = i

                selected_columns_sse_nid = ['Hospital','Pincipal Diagnosis']
                specific_columns = ["Age", "Gender",  "Risk", "Admission Type", "Race"]

                matrix_cipher_sse = functions.generate_submatrix(matrix_cipher, selected_columns_sse_nid)

                volumn_cipher_sse= functions.count_frequency_multicol(np.array(matrix_cipher_sse[1:]), matrix_cipher_sse[0])
                volumn_plain_sse = functions.count_frequency_multicol(np.array(matrix_plain_sse[1:]), matrix_plain_sse[0])

                for column in selected_columns_sse_nid:
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

                    feature_vector_cipher = functions.compute_feature_vector(matrix_cipher, column, "PUDF")
                    feature_vector_plain = functions.compute_feature_vector(matrix_plain, column, "PUDF")

                    sorted_cipher_keys = sorted(freq_cipher_dict.keys())
                    sorted_plain_keys = sorted(freq_plain_dict.keys())

                    # 创建密文key到index的映射
                    cipher_key_to_idx = {k: i for i, k in enumerate(sorted_cipher_keys)}
                    plain_key_to_idx = {k: i for i, k in enumerate(sorted_plain_keys)}
                    recovered_kw = list(value_mapping_id.keys())

                    for keyword, freq_cipher in freq_cipher_dict_inPeriod.items():
                        if keyword not in recovered_kw:
                            cipher_idx = cipher_key_to_idx[keyword]
                            feature_cipher = feature_vector_cipher[keyword]
                            freq_cipher = freq_cipher_dict_inPeriod[keyword]
                            vol_cipher = volume_cipher_dict[keyword]
                            min_score = 100
                            for keyword_plain, freq_plain in freq_plain_dict_inPeriod.items():
                                plain_idx = plain_key_to_idx[keyword_plain]
                                feature_plain = feature_vector_plain[keyword_plain]
                                freq_plain = freq_plain_dict_inPeriod[keyword_plain]
                                vol_plain = volume_plain_dict[keyword_plain]

                                # 计算特征向量的欧氏距离
                                fea_cost = np.linalg.norm(np.array(feature_cipher) - np.array(feature_plain))
                                # 计算频率向量的欧氏距离
                                freq_cost = np.linalg.norm(np.array(freq_cipher) - np.array(freq_plain))
                                # 计算体积（标量）的绝对差
                                vol_cost = (vol_cipher - vol_plain) ** 2
                                total_cost = 10 * fea_cost + freq_cost + vol_cost
                                if total_cost < min_score:
                                    min_score = total_cost
                                    min_keyword = keyword_plain
                            value_mapping_id[keyword] = min_keyword
                            value_mapping[keyword] = min_keyword

                re_c = record_count
                re_s_c = len(value_mapping_id) - len(value_mapping)
                kw_c = keyword_count
                kw_s_c = len(value_mapping)
            #     times += 1

            # kw_c_aver = kw_c // times
            # kw_s_c_aver = kw_s_c // times
            # re_c_aver = re_c // times
            # re_s_c_aver = re_s_c // times

                file_extension = os.path.splitext(file_name)[1]
                new_file_name = file_name.replace(file_extension, ".txt")
                outputPath = os.path.join(out,new_file_name)
                print(outputPath)
                with open(outputPath,"w") as f:
                    f.write("record number: " + str(re_c) + " successfully recovered number: " + str(re_s_c) + " keywords number: " + str(kw_c) + " successfully recovered keyword number: " + str(kw_s_c) + "percent of file" + str(frac))




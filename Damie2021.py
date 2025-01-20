import csv
from tqdm import tqdm
import random
from collections import Counter
import functions
import numpy as np
import os
from collections import defaultdict
import math

def get_sub_matrix(V_cipher, cipher_list, knownQ):
                """
                从矩阵V_cipher中按照knownQ筛选列，获取子矩阵
                :param V_cipher: 原始矩阵，以二维列表形式表示
                :param cipher_list: 关键字列表，对应矩阵行列代表的关键字
                :param knownQ: 用于筛选矩阵列的关键字列表
                :return: 筛选后的子矩阵，同样为二维列表形式
                """
                # 获取列索引
                column_indices = [cipher_list.index(keyword) for keyword in knownQ if keyword in cipher_list]
                sub_matrix = []
                for row in V_cipher:
                    sub_row = [row[index].item() for index in column_indices]
                    sub_matrix.append(sub_row)
                return sub_matrix

index_list = [1,2,3,4]

for il in index_list:
    base = 'F:/Desktop/text/'+ str(il)+ 'q2010/'
    # target = '/Users/cherry/Desktop/zengli/'+ str(il)+ 'q2010plain/'  
    out = 'F:/Desktop/text/new/Damie2021/'+ str(il)+ 'q2010 output of Damie2021/'
    lists = [i[:-4] for i in os.listdir(out)]
    file_list = os.listdir(base)
    for file_name in sorted(file_list):
        if file_name[:-4] not in lists:
            filePath = os.path.join(base,file_name)
            matrix = functions.read_csv_to_matrix(filePath)
            # filePathPlain = "D:\\Users\\Lizi\\Desktop\\text\\2q2010\\text_450950.csv"
            filePathPlain = "2015/PUDF_base1_1q2015.csv"
            matrixP = functions.read_csv_to_matrix(filePathPlain)

            selected_columns = ['Record ID', 'Age', 'Admission Type', 'Length of stay', 'Risk', 'Gender', 'Race','Hospital', 'Pincipal Diagnosis']
            selected_columns_noid = ['Age', 'Admission Type', 'Length of stay', 'Risk', 'Gender', 'Race','Hospital', 'Pincipal Diagnosis']

            matrix_cipher = functions.generate_submatrix(matrix, selected_columns)
            matrix_plain = functions.generate_submatrix(matrixP, selected_columns)

            # 针对OPE的攻击，首先判断每一列的数据种类，如果|C| = |M|那么可以直接排序
            selected_column_ope = ['Record ID', 'Age', 'Admission Type', 'Length of stay', 'Risk']
            matrix_cipher_ope = functions.generate_submatrix(matrix_cipher, selected_column_ope)
            matrix_plain_ope = functions.generate_submatrix(matrix_plain, selected_column_ope)
            value_mapping = {}
            # Age
            column_cipher_age = functions.extract_columns(matrix_cipher_ope, 'Age')
            age_cipher_count = Counter(column_cipher_age)
            keyword_count_cipher_age = len(age_cipher_count)

            column_plain_age = functions.extract_columns(matrix_plain_ope, 'Age')
            age_plain_count = Counter(column_plain_age)
            keyword_count_plain_age = len(age_plain_count)
            value_mapping = {}

            if keyword_count_cipher_age == keyword_count_plain_age:
                sorted_cipher_age = sorted(set(column_cipher_age))
                sorted_plain_age = sorted(set(column_plain_age))
                mapping_ope_age = {key: value for key, value in zip(sorted_cipher_age, sorted_plain_age)}
                value_mapping.update(mapping_ope_age)
            else:
                mapping_ope_age = functions.find_optimal_mapping(column_cipher_age, column_plain_age)
                value_mapping.update(mapping_ope_age)

            # Admission Type
            column_cipher_admi = functions.extract_columns(matrix_cipher_ope, 'Admission Type')
            admi_cipher_count = Counter(column_cipher_admi)
            keyword_count_cipher_admi = len(admi_cipher_count)

            column_plain_admi = functions.extract_columns(matrix_plain_ope, 'Admission Type')
            admi_plain_count = Counter(column_plain_admi)
            keyword_count_plain_admi = len(admi_plain_count)

            if keyword_count_cipher_admi == keyword_count_plain_admi:
                sorted_cipher_admi = sorted(set(column_cipher_admi))
                sorted_plain_admi = sorted(set(column_plain_admi))
                mapping_ope_admi = {key: value for key, value in zip(sorted_cipher_admi, sorted_plain_admi)}
                value_mapping.update(mapping_ope_admi)
            else:
                mapping_ope_admi = functions.find_optimal_mapping(column_cipher_admi, column_plain_admi)
                value_mapping.update(mapping_ope_admi)

            # Risk
            column_cipher_risk = functions.extract_columns(matrix_cipher_ope, 'Risk')
            risk_cipher_count = Counter(column_cipher_risk)
            keyword_count_cipher_risk = len(risk_cipher_count)

            column_plain_risk = functions.extract_columns(matrix_plain_ope, 'Risk')
            risk_plain_count = Counter(column_plain_risk)
            keyword_count_plain_risk = len(risk_plain_count)

            if keyword_count_cipher_risk == keyword_count_plain_risk:
                sorted_cipher_risk = sorted(set(column_cipher_risk))
                sorted_plain_risk = sorted(set(column_plain_risk))
                mapping_ope_risk = {key: value for key, value in zip(sorted_cipher_risk, sorted_plain_risk)}
                value_mapping.update(mapping_ope_risk)
            else:
                mapping_ope_risk = functions.find_optimal_mapping(column_cipher_risk, column_plain_risk)
                value_mapping.update(mapping_ope_risk)

            # Length of Stay
            column_cipher_stay = functions.extract_columns(matrix_cipher_ope, 'Length of stay')
            stay_cipher_count = Counter(column_cipher_stay)
            keyword_count_cipher_stay = len(stay_cipher_count)

            column_plain_stay = functions.extract_columns(matrix_plain_ope, 'Length of stay')
            stay_plain_count = Counter(column_plain_stay)
            keyword_count_plain_stay = len(stay_plain_count)

            if keyword_count_cipher_stay == keyword_count_plain_stay:
                sorted_cipher_stay = sorted(set(column_cipher_stay))
                sorted_plain_stay = sorted(set(column_plain_stay))
                mapping_ope_stay = {key: value for key, value in zip(sorted_cipher_stay, sorted_plain_stay)}
                value_mapping.update(mapping_ope_stay)
            else:
                sorted_cipher_stay = sorted(column_cipher_stay, key=functions.custom_sort)
                sorted_plain_stay = sorted(column_plain_stay, key=functions.custom_sort)
                mapping_ope_stay = functions.find_optimal_mapping(sorted_cipher_stay, sorted_plain_stay)
                value_mapping.update(mapping_ope_stay)

            keyword_count_stay = functions.count_keywords(functions.generate_submatrix(matrix_cipher,[
                'Length of stay'
                # ,'Age', 'Admission Type', 'Risk'
                # ,'Discharge', 'Gender', 'Race'
                ])[1:]) 

            keyword_count_aar = functions.count_keywords(functions.generate_submatrix(matrix_cipher,[
                # 'Length of stay',
                'Age', 'Admission Type', 'Risk'
                # ,'Discharge', 'Gender', 'Race'
                ])[1:]) 
            print(f"length of stay: {keyword_count_stay}")
            print(f"Age + Admission Type + Risk: {keyword_count_aar}")

            # DET的部分，计算频率和种类
            selected_columns_det = ['Gender', 'Race']
            matrix_cipher_det = functions.generate_submatrix(matrix_cipher, selected_columns_det)
            matrix_plain_det = functions.generate_submatrix(matrix_plain, selected_columns_det)

            # 计算频率
            element_cipher_det = functions.column_frequencies(matrix_cipher_det[1:])
            element_plain_det = functions.column_frequencies(matrix_plain_det[1:])

            result = {}
            for column, dict1 in element_cipher_det.items():
                dict2 = element_plain_det[column]
                temp = functions.find_closest_mapping(dict1,dict2)
                result.update(temp)
            print(f"DET的关键字个数为：{len(result)}")

            for key, dict1 in element_cipher_det.items():
                dict2 = element_plain_det[key]
                temp = functions.find_closest_mapping(dict1, dict2)
                value_mapping.update(temp)
            
            value_mapping_od = {}
            for key, value in value_mapping.items():
                if key == value:
                    value_mapping_od[key] = value

            selected_columns_od = ['Record ID', 'Age', 'Admission Type', 'Length of stay', 'Risk','Gender', 'Race']

            matrix_cipher_od = functions.generate_submatrix(matrix_cipher, selected_columns_od)  # 经过OPE和DET后恢复了多少个记录
            matrix_recovered = functions.replace_values_with_none(matrix_cipher_od[1:], value_mapping_od)

            matrix_plain_od = functions.generate_submatrix(matrix_plain, selected_columns_od)
            list_cipher_od = []
            record_cipher = []
            for row in matrix_recovered[1:]:
                if None not in row[1:]:
                    res = ' '.join(row[1:])
                    record_cipher.append(row[0])
                    list_cipher_od.append(res)
            frequency_cipher_od  = Counter(list_cipher_od)

            list_plain_od = []
            record_plain = []
            for row in matrix_plain_od[1:]:
                res = ' '.join(row[1:])
                record_plain.append(row[0])
                list_plain_od.append(res)
            frequency_plain_od = Counter(list_plain_od)

            unique_values_cipher_od = functions.find_unique_value(list_cipher_od)
            unique_values_plain_od = functions.find_unique_value(list_plain_od)

            unique_rows = [v for v in unique_values_cipher_od if v in unique_values_plain_od]

            recovered_rows = {}
            for row in tqdm(unique_rows, desc="Processing rows", total=len(unique_rows)):
                id_cipher = record_cipher[list_cipher_od.index(row)]
                id_plain = record_plain[list_plain_od.index(row)]
                recovered_rows[id_cipher] = id_plain

            print(len(recovered_rows))

            recordid_cipher = functions.create_rowid_dict(matrix_cipher, matrix_cipher[0], 'Record ID')
            recordid_plain = functions.create_rowid_dict(matrix_plain, matrix_plain[0], 'Record ID')

            for id_c, id_p in recovered_rows.items():
                # index_cipher = recordid_cipher[id_c]
                row_cipher = matrix_cipher[recordid_cipher[id_c]]
                index_plain = recordid_plain[id_p]
                row_plain = matrix_plain[index_plain]
                for i in range(1,len(row_cipher)):
                    value_mapping[row_cipher[i]] = row_plain[i]

            for key, value in value_mapping.items():
                if key == value:
                    value_mapping_od[key] = value

            # SSE
            print("------sse-----")
            selected_column_sse = ['Hospital','Pincipal Diagnosis']
            matrix_cipher_sse = functions.generate_submatrix(matrix_cipher, selected_column_sse)
            # Adversary Information

            # 1. Similar Document
            matrix_plain_sse = functions.generate_submatrix(matrix_plain, selected_column_sse)

            # 2. Vocabulary kw_plain of auxilary information
            hospital_plain = functions.extract_columns(matrix_plain, 'Hospital')
            hospital_plain_list = list(set(hospital_plain))
            diagnosis_plain = functions.extract_columns(matrix_plain, 'Pincipal Diagnosis')
            diagnosis_plain_list = list(set(diagnosis_plain))
            plain_list = list(set(hospital_plain_list + diagnosis_plain_list))

            # 3. 共现矩阵：两个关键字出现在同一个文档中的次数
            Nd_plain = len(matrix_plain[1:])  # 文档总数

            # 创建一个与矩阵数据部分相同形状的全 False 数组
            keyword_matrix = np.zeros((Nd_plain, len(plain_list)), dtype=bool)

            # 填充 keyword_matrix，标记每个关键字在哪些文档中出现
            for i, keyword in tqdm(enumerate(plain_list), total=len(plain_list)):
                keyword_matrix[:, i] = np.array([keyword in row for row in matrix_plain_sse[1:]])

            # 计算 V 矩阵
            V_plain = np.einsum('ij,ik->jk', keyword_matrix, keyword_matrix) / Nd_plain

            # 使 V 矩阵对称
            V_plain = (V_plain + V_plain.T) / 2

            # 4. Vocabulary kw_cipher 
            hospital_cipher = functions.extract_columns(matrix_cipher, 'Hospital')
            hospital_cipher_list = list(set(hospital_cipher))
            diagnosis_cipher = functions.extract_columns(matrix_cipher, 'Pincipal Diagnosis')
            diagnosis_cipher_list = list(set(diagnosis_cipher))
            cipher_list = list(set(hospital_cipher_list + diagnosis_cipher_list))

            # 5. 密文的共现矩阵：两个关键字出现在同一个文档中的次数
            Nd_cipher = len(matrix_cipher[1:])  # 文档总数

            keyword_matrix = np.zeros((Nd_cipher, len(cipher_list)), dtype=bool)

            # 使用tqdm添加进度条
            for i, keyword in tqdm(enumerate(cipher_list), total=len(cipher_list)):
                keyword_matrix[:, i] = np.array([keyword in row for row in matrix_cipher_sse[1:]])

            V_cipher = np.einsum('ij,ik->jk', keyword_matrix, keyword_matrix) / Nd_cipher
            V_cipher = (V_cipher + V_cipher.T) / 2

            # 随机选择15%的关键字作为knownQ
            knownQ = random.sample(cipher_list, int(0.15 * len(cipher_list)))

            # 未知查询和已知查询在同一个文档中的频率
            C_td = get_sub_matrix(V_cipher, cipher_list, knownQ)
            # 关键字集合中关键字和已知关键字在同一个文档中的频率
            C_kw = get_sub_matrix(V_plain, plain_list, knownQ)
            # print(len(C_td))
            # print(len(C_td[0]))
            # print(len(C_kw))
            # print(len(C_kw[0]))
            RefSpeed = 10
            knownQ_dict = [(kw, kw) for kw in knownQ]
            unknownQ = [elem for elem in cipher_list if elem not in knownQ]

            print("knownQ origin: "+ str(len(knownQ)))
            print("unknownQ origin: "+str(len(unknownQ)))

            while unknownQ:
                final_pred = []
                unknownQ_new = [elem for elem in cipher_list if elem not in knownQ]
                temp_pred = []

                # 提前构建索引字典
                plain_index_dict = {kw: index for index, kw in enumerate(plain_list)}
                cipher_index_dict = {td: index for index, td in enumerate(cipher_list)}

                for td in unknownQ_new:
                    cand = []
                    td_index = cipher_index_dict[td]
                    v2 = C_td[td_index]
                    for kw in plain_list:
                        kw_index = plain_index_dict[kw]
                        v1 = C_kw[kw_index]
                        # 使用numpy计算欧氏距离
                        euclidean_distance = sum((a - b) ** 2 for a, b in zip(v1, v2))
                        if euclidean_distance == 0:
                            s = 0
                        else:
                            s = -math.log(euclidean_distance)
                        cand.append({"kw": kw, "score": s})
                    cand.sort(key=lambda x: x["score"], reverse=True)
                    certainty = cand[0]["score"] - cand[1]["score"]
                    temp_pred.append((td, cand[0]["kw"], certainty))

                if len(unknownQ) < RefSpeed:
                    final_pred.extend(temp_pred)
                    unknownQ = []
                    break
                else:
                    most_certain = sorted(temp_pred, key=lambda x: x[2], reverse=True)[:RefSpeed]
                    for td, kw, _ in most_certain:
                        knownQ_dict.append((td, kw))
                        unknownQ = list(filter(lambda x: x!= td, unknownQ))
                        knownQ.append(td)
                    # print("knownQ: "+ str(len(knownQ)))
                    unknownQ_new = [elem for elem in cipher_list if elem not in knownQ]
                    # 未知查询和已知查询在同一个文档中的频率
                    C_td = get_sub_matrix(V_cipher, cipher_list, knownQ)
                    # 关键字集合中关键字和已知关键字在同一个文档中的频率
                    C_kw = get_sub_matrix(V_plain, plain_list, knownQ)
                    # print("unknownQ: "+str(len(unknownQ)))

            count = 0
            for row in final_pred:
                key = row[0]
                value = row[1]
                if key == value:
                    value_mapping_od[key] = value
                    count += 1

            print(count)
            print(len(value_mapping_od))
            keyword_number = functions.count_keywords(functions.generate_submatrix(matrix_cipher, selected_columns_noid)[1:])

            file_extension = os.path.splitext(file_name)[1]
            new_file_name = file_name.replace(file_extension, ".txt")
            outputPath = os.path.join(out,new_file_name)
            print(outputPath)
            with open(outputPath,"w") as f:
                f.write("keywords number: " + str(keyword_number) + " successfully recovered keyword number: " + str(len(value_mapping_od)))


                

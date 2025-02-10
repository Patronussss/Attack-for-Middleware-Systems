import functions
import os
from collections import Counter
from tqdm import tqdm
import numpy as np

index_list = [1]
frac_list = [0.1, 0.3, 0.5, 0.7, 0.9]

for frac in frac_list:
    for il in index_list:
        base = str(il)+ 'q2010/'
        out = str(il)+ 'q2010 output of A3/' + str(frac) + '/'

        if not os.path.exists(out):
            os.makedirs(out)

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
            matrix = functions.read_csv_to_matrix(filePath)

            selected_columns_cipher = ['Record ID', 'Age', 'Admission Type', 'Length of stay', 'Risk','Discharge', 'Gender', 'Race','Hospital', 'Pincipal Diagnosis']

            matrix_cipher = functions.generate_submatrix(matrix, selected_columns_cipher)
            num_rows_to_extract =  int((len(matrix_cipher)-1) * frac)

            matrix_plain = functions.random_submatrix(matrix_cipher, num_rows_to_extract)
            record_count = len(matrix_plain)-1 #能够过恢复的最多的record 记录（剩下的没有对照的identifire就算恢复了所有数据也没法比较）
            keyword_count = functions.count_keywords(functions.generate_submatrix(matrix_cipher,['Age', 'Admission Type', 'Length of stay', 'Risk','Discharge', 'Gender', 'Race','Hospital', 'Pincipal Diagnosis'])[1:]) #能够恢复的关键字个数，由于只能拿到部分明文数据库，因此能够拿到的关键字个数和明文数据库拥有的明文个数相关

            # 针对OPE的攻击，排序和CDF
            # 同时比较CDF和频率，找到一一对应关系
            selected_column_ope = ['Record ID', 'Age', 'Admission Type', 'Length of stay', 'Risk']
            matrix_cipher_ope = functions.generate_submatrix(matrix_cipher, selected_column_ope)
            matrix_plain_ope = functions.generate_submatrix(matrix_plain, selected_column_ope)

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

            # print(value_mapping)
            keyword_count_ope_det = functions.count_keywords(functions.generate_submatrix(matrix_plain,[
                'Length of stay'
                ,'Age', 'Admission Type', 'Risk'
                ,'Discharge', 'Gender', 'Race'
                ])[1:]) 
            print(f"OPE+DET一共包含{keyword_count_ope_det}个关键字")

            keyword_count_ope = functions.count_keywords(functions.generate_submatrix(matrix_plain,[
                'Length of stay'
                ,'Age', 'Admission Type', 'Risk'
                # ,'Discharge', 'Gender', 'Race'
                ])[1:]) 
            print(f"OPE一共包含{keyword_count_ope}个关键字")

            keyword_count_ope_nostay = functions.count_keywords(functions.generate_submatrix(matrix_plain,[
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
            matrix_cipher_det = functions.generate_submatrix(matrix_cipher, selected_column_det)
            matrix_plain_det = functions.generate_submatrix(matrix_plain, selected_column_det)

            # 计算频率
            element_cipher_det = functions.column_frequencies(matrix_cipher_det[1:])
            element_plain_det = functions.column_frequencies(matrix_plain_det[1:])

            result = {}
            for key, dict1 in element_cipher_det.items():
                dict2 = element_plain_det[key]
                temp = functions.find_closest_mapping(dict1, dict2)
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

            selected_columns_od = ['Record ID', 'Age', 'Admission Type', 'Risk','Discharge', 'Gender', 'Race']
            matrix_plain_od = functions.generate_submatrix(matrix_plain, selected_columns_od)
            matrix_cipher_od = functions.generate_submatrix(matrix_cipher, selected_columns_od)
            matrix_recovered_od = functions.replace_values_with_none(matrix_cipher_od, value_mapping_od)
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

            print(len(recovered_rows))
            count = 0 
            for key,value in recovered_rows.items():
                if key == value:
                    count += 1
            print(count)

            recordid_cipher = functions.create_rowid_dict(matrix_cipher, matrix_cipher[0], 'Record ID')
            recordid_plain = functions.create_rowid_dict(matrix_plain, matrix_plain[0], 'Record ID')

            for id_c, id_p in recovered_rows.items():
                # index_cipher = recordid_cipher[id_c]
                row_cipher = matrix_cipher[recordid_cipher[id_c]]
                index_plain = recordid_plain[id_p]
                row_plain = matrix_plain[index_plain]
                for i in range(1, len(row_cipher)):
                    value_mapping_od[row_cipher[i]] = row_plain[i]

            selected_columns_cipher_iod = ['Record ID', 'Age', 'Admission Type', 'Length of stay', 'Risk','Discharge', 'Gender', 'Race']

            matrix_cipher_iod = functions.generate_submatrix(matrix_cipher, selected_columns_cipher_iod)  # 经过OPE和DET后恢复了多少个记录
            matrix_recovered = functions.replace_values_with_none(matrix_cipher_iod, value_mapping_od)

            record_recovered = functions.find_unique_rows_withNone(matrix_recovered)
            print(f"经过OPE+DET一共找到了{len(record_recovered)}条数据")

            # SSE
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
                value_mapping_od.update(common_keywords)
                # print(len(value_mapping))

                keyword_count_sse = functions.count_keywords(functions.generate_submatrix(matrix_plain,[
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

            matrix_cipher_iods = functions.generate_submatrix(matrix_cipher, selected_columns_cipher_iods)  # 经过OPE和DET后恢复了多少个记录
            matrix_recovered_sse = functions.replace_values_with_none(matrix_cipher_iods, value_mapping_ods)

            record_recovered_iods = functions.find_unique_rows_withNone(matrix_recovered_sse)

            # 创建字典记录行号和对应的record id列的值
            record_plain = [row[0] for row in matrix_plain]
            record = list(set(record_plain) & set(record_recovered_iods))

            if len(record) == 0 and len(matrix_plain) > 5000:
                print("len(record) = 0")
                continue

            print(f"成功恢复了{len(record)}条数据")
            print(f"一共拥有{len(matrix_plain)}条记录")
            print(f"恢复记录比例：{len(record) / len(matrix_plain)}")

            row_count = {}
            for row in matrix_plain:
                row_data = tuple(row[1:])  # 从第二列开始组合
                if row_data in row_count:
                    row_count[row_data] += 1
                else:
                    row_count[row_data] = 1

            unique_rows = [list(row) for row, count in row_count.items() if count == 1]
            print(f"plain中一共有{len(unique_rows)}条数据是独特的，也就是说理论上可以恢复的数据大小")

            # SSE
            # Step 1：尝试使用volume leakage，即寻找每一列中匹配记录频率独特的关键字
            selected_column_sse = ['Hospital','Pincipal Diagnosis']
            matrix_cipher_sse = functions.generate_submatrix(matrix_cipher, selected_column_sse)
            matrix_plain_sse = functions.generate_submatrix(matrix_plain, selected_column_sse)

            # 每个关键字的volume
            volumn_cipher_sse = functions.count_frequency_multicol(np.array(matrix_cipher_sse[1:]), matrix_cipher_sse[0])
            volumn_plain_sse = functions.count_frequency_multicol(np.array(matrix_plain_sse[1:]), matrix_plain_sse[0])
            mapping_sse = {}
            for key, dict1 in volumn_cipher_sse.items():
                dict2 = volumn_plain_sse[key]
                temp = functions.find_closest_mapping(dict1, dict2)
                value_mapping.update(temp)
                mapping_sse.update(temp)

            for key, value in mapping_sse.items():
                if key == value:
                    value_mapping_od[key] = value
                
            # keyword_number = count_keywords(generate_submatrix(matrix_cipher, selected_columns_noid)[1:])

            hospital_plain = functions.extract_columns(matrix_plain, 'Hospital')
            hospital_cipher = functions.extract_columns(matrix_cipher, 'Hospital')
            hospital_list = list(set(hospital_plain).union(set(hospital_cipher)))
            diagnosis_plain = functions.extract_columns(matrix_plain, 'Pincipal Diagnosis')
            diagnosis_cipher = functions.extract_columns(matrix_cipher, 'Pincipal Diagnosis')

            diagnosis_list = list(set(diagnosis_plain).union(set(diagnosis_cipher)))
            list_dh = hospital_list + diagnosis_list
            hospital_frequency_dict = {}
            frequency_folder = 'frequency/'
            for value in hospital_list:
                csv_file_path = os.path.join(frequency_folder, f'{value}.csv')
                if os.path.exists(csv_file_path):
                    value_dict = functions.process_csv_file(csv_file_path)
                    hospital_frequency_dict[value] = value_dict

            # 每个关键字的搜索频率
            hos_freq_cipher_dict = {key: value for key,value in hospital_frequency_dict.items() if key in hospital_cipher}
            hos_freq_plain_dict = {key: value for key,value in hospital_frequency_dict.items() if key in hospital_plain}
            diagnosis_frequency_dict = {}

            for value in diagnosis_list:
                csv_file_path = os.path.join(frequency_folder, f'{value}.csv')
                if os.path.exists(csv_file_path):
                    value_dict = functions.process_csv_file(csv_file_path)
                    diagnosis_frequency_dict[value] = value_dict

            # 每个关键字的搜索频率
            diag_freq_cipher_dict = {key: value for key,value in diagnosis_frequency_dict.items() if key in diagnosis_cipher}
            diag_freq_plain_dict = {key: value for key,value in diagnosis_frequency_dict.items() if key in diagnosis_plain}

            # Step 1 计算v = 匹配上的文档数/总文档数
            hos_volume_cipher_dict = volumn_cipher_sse['Hospital']
            diag_volume_cipher_dict = volumn_cipher_sse['Pincipal Diagnosis']
            hos_volume_plain_dict = volumn_plain_sse['Hospital']
            diag_volume_plain_dict = volumn_plain_sse['Pincipal Diagnosis']

            hos_freq_cipher_dict1, hos_freq_cipher_dict_noT = functions.numToFreqency(hos_freq_cipher_dict)
            hos_freq_plain_dict1, hos_freq_plain_dict_noT = functions.numToFreqency(hos_freq_plain_dict)
            diag_freq_cipher_dict1, diag_freq_cipher_dict_noT = functions.numToFreqency(diag_freq_cipher_dict)
            diag_freq_plain_dict1, diag_freq_plain_dict_noT = functions.numToFreqency(diag_freq_plain_dict)

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
                                f1 = freq1[functions.data[i]]
                                f2 = freq2[functions.data[i]]
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
                                f1 = freq1[functions.data[i]]
                                f2 = freq2[functions.data[i]]
                                f += abs(f1-f2)
                            f = f/len(freq1) * (1-alpha)
                            score = v+f
                            if score < min_score:
                                min_score = score
                    dis_dict[key1] = min_score

            sorted_dis_dict = dict(sorted(dis_dict.items(), key=lambda item: item[1], reverse=True))
            # 对上述查询，找关键字
            pred_dict = {}
            count = 0
            for key1, value in sorted_dis_dict.items():
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
                                f1 = freq1[functions.data[i]]
                                f2 = freq2[functions.data[i]]
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
                                f1 = freq1[functions.data[i]]
                                f2 = freq2[functions.data[i]]
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
            matrix_recovered = functions.replace_values_with_none(matrix_cipher, value_mapping_ods)

            record_recovered = functions.find_unique_rows_withNone(matrix_recovered)
            print(f"经过OPE+DET+SSE一共找到了{len(record_recovered)}条数据")

            re_c += record_count
            re_s_c += len(record_recovered)
            kw_c += keyword_count
            kw_s_c += len(value_mapping_ods)
            times += 1

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

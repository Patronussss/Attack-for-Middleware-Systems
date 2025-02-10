import functions
import os
from collections import Counter
from tqdm import tqdm

index_list = [1,2,3,4]

for il in index_list:
    base = str(il)+ 'q2010/'
    out = str(il)+ 'q2010 output of A4/'
    lists = [i[:-4] for i in os.listdir(out)]
    file_list = os.listdir(base)
    for file_name in sorted(file_list):
        if file_name[:-4] not in lists:
            filePath = os.path.join(base,file_name)
            matrix = functions.read_csv_to_matrix(filePath)
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
                mapping_ope_age = functions.find_optimal_mapping_A4(column_cipher_age, column_plain_age)
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
                mapping_ope_admi = functions.find_optimal_mapping_A4(column_cipher_admi, column_plain_admi)
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
                mapping_ope_risk = functions.find_optimal_mapping_A4(column_cipher_risk, column_plain_risk)
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
                mapping_ope_stay = functions.find_optimal_mapping_A4(sorted_cipher_stay, sorted_plain_stay)
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
                
            keyword_number = functions.count_keywords(functions.generate_submatrix(matrix_cipher, selected_columns_noid)[1:])

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

            pred_dict = {}
            count = 0
            for key1, value in sorted_dis_dict.items():
                # count += 1
                # if count == 500:
                #     break
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
            count = 0
            for key,value in pred_dict.items():
                if key == value:
                    value_mapping_od[key] = value
                    count += 1
            print(count)
            print(len(value_mapping_od))
        
            file_extension = os.path.splitext(file_name)[1]
            new_file_name = file_name.replace(file_extension, ".txt")
            outputPath = os.path.join(out,new_file_name)
            print(outputPath)
            with open(outputPath,"w") as f:
                f.write("keywords number: " + str(keyword_number) + " successfully recovered keyword number: " + str(len(value_mapping_od)))
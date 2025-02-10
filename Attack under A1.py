import functions
import os 
import csv
import random
from tqdm import tqdm
from collections import Counter

index_list = [1,2,3,4]

for il in index_list:
    base = str(il) + 'q2010/'
    target = str(il) + 'q2010/'
    out = str(il) + 'q2010 outpuut of A1'
    lists = [i[:-4] for i in os.listdir(out)]
    file_list = os.listdir(base)
    for file_name in sorted(file_list):
        filePath = os.path.join(base,file_name)
        filePathPlain = os.path.join(target,file_name)

        matrix_cipher = functions.read_csv_to_matrix(filePath)
        record_id_mapping = functions.create_rowid_dict(matrix_cipher, matrix_cipher[0], 'Record ID') # 记录行置换情况

        selected_columns_ope_det_sse_withoutid = [ 'Age', 'Admission Type', 'Length of stay', 'Risk','Discharge', 'Gender', 'Race','Hospital', 'Pincipal Diagnosis']
        matrix_ods_withoutid = functions.generate_submatrix(matrix_cipher, matrix_cipher[0], selected_columns_ope_det_sse_withoutid)

        selected_columns_ope_det_sse = ['Record ID', 'Age', 'Admission Type', 'Length of stay', 'Risk','Discharge', 'Gender', 'Race','Hospital', 'Pincipal Diagnosis']
        matrix_cipher_ods = functions.generate_submatrix(matrix_cipher, matrix_cipher[0], selected_columns_ope_det_sse)

        keyword_count = functions.count_keywords(matrix_ods_withoutid[1:]) # 一共有多少个关键字
        record_count = len(matrix_cipher) - 1 # 一共有多少条数据

        # 计算OPE+DET的关键字数量
        selected_columns_ope = ['Record ID', 'Age', 'Admission Type', 'Length of stay', 'Risk']
        matrix_cipher_ope = functions.generate_submatrix(matrix_cipher, matrix_cipher[0], selected_columns_ope)
        def custom_sort(row):
            return tuple(int(cell) if cell.isdigit() else cell for cell in row[1:])

        # 按照要求对矩阵进行排序
        sorted_matrix_cipher = sorted(matrix_cipher_ope, key=custom_sort)
        # 读取原始CSV文件
        with open(filePath, 'r') as infile:
            reader = csv.reader(infile)
            header = next(reader)  # 保留表头
            data = [row for row in reader]

        # 随机打乱行顺序
        random.shuffle(data)

        with open(filePathPlain, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)  # 写入表头
            writer.writerows(data)

        matrix_plain = functions.read_csv_to_matrix(filePathPlain)
        record_id_mapping_plain = functions.create_rowid_dict(matrix_plain, matrix_plain[0], 'Record ID') # 记录行置换情况

        matrix_plain_ope = functions.generate_submatrix(matrix_plain, matrix_plain[0], selected_columns_ope)

        # 按照要求对矩阵进行排序
        sorted_matrix_plain = sorted(matrix_plain_ope, key=custom_sort)

        value_mapping = {}

        for i in range(len(sorted_matrix_cipher)):
            row_cipher = sorted_matrix_cipher[i][1:]  # 去掉第一列
            row_plain = sorted_matrix_plain[i][1:]   # 去掉第一列
            if row_cipher == row_plain:
                # matching_pairs.append((sorted_matrix_cipher[i][0], sorted_matrix_plain[i][0]))  # 记录第一列的值
                for j in range(len(row_cipher)):
                    value_mapping[row_cipher[j]] = row_plain[j]

        selected_columns_det = ['Record ID', 'Discharge', 'Gender', 'Race']
        matrix_cipher_det = functions.generate_submatrix(matrix_cipher, matrix_cipher[0], selected_columns_det)

        element_counts = {}  # 用于存储元素及其出现次数的字典
        for col_index in range(1, len(matrix_cipher_det[0])):  # 从第二列开始（索引为1）
            column_data = [row[col_index] for row in matrix_cipher_det]  # 提取该列的数据
            for element in column_data:
                if element in element_counts:
                    element_counts[element] += 1
                else:
                    element_counts[element] = 1

        matrix_plain_det = functions.generate_submatrix(matrix_plain, matrix_plain[0], selected_columns_det)

        element_counts_plain = {}  # 用于存储元素及其出现次数的字典
        for col_index in range(1, len(matrix_plain_det[0])):  # 从第二列开始（索引为1）
            column_data = [row[col_index] for row in matrix_plain_det]  # 提取该列的数据
            for element in column_data:
                if element in element_counts_plain:
                    element_counts_plain[element] += 1
                else:
                    element_counts_plain[element] = 1
        # print(element_counts_plain)

        for key1, value1 in element_counts.items():
            for key2, value2 in element_counts_plain.items():
                if value1 == value2:
                    value_mapping[key1] = key2

        selected_columns_ope_det = ['Record ID','Age', 'Admission Type', 'Length of stay', 'Risk','Discharge', 'Gender', 'Race']
        matrix_cipher_ope_det = functions.generate_submatrix(matrix_cipher, matrix_cipher[0], selected_columns_ope_det)

        keyword_count_ope_det = len(value_mapping)

        matrix_od_replaced = functions.replace_values_with_none(matrix_cipher_ope_det, value_mapping)
        unique_rows_replaced = functions.find_unique_rows_withNone(matrix_od_replaced)

        # 开始SSE

        selected_columns_sse = ['Record ID', 'Hospital','Pincipal Diagnosis']
        data_dict_cipher = {}
        stop_words = ["and", "of", "or", "for", "with", "to", "not", "by", "in", "the", "but", "from", "as"]

        matrix_cipher_sse = functions.generate_submatrix(matrix_cipher, matrix_cipher[0], selected_columns_sse)

        for row in matrix_cipher_sse:
            record_id = row[0]
            for i in range(1, len(row)):
                word = row[i]
                if word.lower() not in stop_words:
                    if word in data_dict_cipher:
                        data_dict_cipher[word].append(record_id)
                    else:
                        data_dict_cipher[word] = [record_id]
        
        matrix_recovered = []
        total_iterations = len(matrix_cipher_sse)
        with tqdm(total=total_iterations, desc="generating matrix recovered") as pbar:
            for row in matrix_cipher_sse:
                if row[0] in unique_rows_replaced:
                    matrix_recovered.append(row)
                pbar.update(1)
        
        # 生成两个十进制序列及其对应的关键字
        decimal_sequence1, keywords1 = functions.generate_decimal_sequence(matrix_cipher_sse)
        decimal_sequence2, keywords2 = functions.generate_decimal_sequence(matrix_cipher_sse)

        # 创建一个映射，将十进制值映射回关键字
        decimal_to_keyword1 = {decimal: keyword for decimal, keyword in zip(decimal_sequence1, keywords1)}
        decimal_to_keyword2 = {decimal: keyword for decimal, keyword in zip(decimal_sequence2, keywords2)}

        # 找出每个序列中独特的值（只出现一次的值）
        unique_values1 = [decimal for decimal in decimal_sequence1 if decimal_sequence1.count(decimal) == 1]
        unique_values2 = [decimal for decimal in decimal_sequence2 if decimal_sequence2.count(decimal) == 1]

        # 找出两个序列中相同的独特值
        common_unique_values = set(unique_values1) & set(unique_values2)

        # 记录这些相同的值对应的关键字
        common_keywords = {decimal_to_keyword1[value]: decimal_to_keyword2[value] for value in common_unique_values if value in decimal_to_keyword1 and value in decimal_to_keyword2}
        value_mapping.update(common_keywords)

        keyword_count = functions.count_keywords(matrix_cipher_ods)-record_count
        
        matrix_s_replaced = functions.replace_values_with_none(matrix_cipher_ods, value_mapping)
        unique_rows_s_replaced = functions.find_unique_rows_withNone(matrix_s_replaced)

        unique = functions.ind_unique_rows(matrix_cipher_ods)

        keyword_volume = {}
        for key, value in data_dict_cipher.items():
            keyword_volume[key] = len(value)

        value_counts = Counter(keyword_volume.values())
        unique_values = [v for v, count in value_counts.items() if count == 1]
        _ = [value_mapping.setdefault(value, [value]) for value in unique_values]
        values = {}
        for key, value in value_mapping.items():
            if key == value:
                values[key] = value

        file_extension = os.path.splitext(file_name)[1]
        new_file_name = file_name.replace(file_extension, ".txt")
        outputPath = os.path.join(out,new_file_name)

        with open(outputPath,"w") as f:
            f.write("record number: " + str(record_count) + " successfully recovered number: " + str(len(unique)) + " keywords number: " + str(keyword_count) + " successfully recovered keyword number: " + str(len(values)) + "SSE+OPE: " + str(keyword_count_ope_det))

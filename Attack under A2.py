import functions
import os

index_list = [1,2,3,4]
for il in index_list:
    file = [500, 1000, 2000, 5000, 7500, 10000, 20000, 50000, 75000, 100000, 120000, 150000, 175000, 200000, 220000,
            250000, 275000, 300000, 325000, 350000, 375000,400000, 425000, 450000]
    if il == 1:
        base = '1q2010/text_538066.csv'
    elif il == 2:
        base = '2q2010/text_450950.csv'
    elif il == 3:
        base = '3q2010/text_496585.csv'
    elif il == 4:
        base = '4q2010/text_508029.csv'
    out = 'output of A2/'

    if not os.path.exists(out):
        os.makedirs(out)

    filePathPlain = "2015/PUDF_base1_1q2015.csv" # auxiliary
    matrixP = functions.read_csv_to_matrix(filePathPlain)
    length = len(functions.read_csv_to_matrix(base))-1
    file.append(length)
    if length > 475000 and length < 500000:
        file.append(475000)
    if length > 500000:
        file.append(475000)
        file.append(500000)

    column_count = {} # 记录每一个数据库中有多少个需要恢复的列
    recovered_count = {} # 成功恢复的列数
    OPEdict = {}
    DETdict = {}

    for f in file:
        flag = 1
        FinalResult = []
        for _ in range(10):
            matrix = functions.select_rows_and_generate_matrix(f, base)
            selected_columns_cipher = ['Age', 'Admission Type', 'Length of stay', 'Risk','Discharge', 'Gender', 'Race','Hospital', 'Pincipal Diagnosis']
            selected_columns_plain = ['Age', 'Admission Type', 'Length of stay', 'Risk','Discharge', 'Gender', 'Race']
            # 构建明文数据库和密文数据库

            matrix_temp = functions.generate_submatrix(matrix, selected_columns_cipher)
            matrix_plain = functions.generate_submatrix(matrixP, selected_columns_plain)

            matrix_cipher =  functions.shuffle_matrix(matrix_temp)

            unique_elements_cipher = functions.count_column_elements(matrix_cipher)
            unique_elements_plain = functions.count_column_elements(matrix_plain)

            unique_counts_cipher = {}
            for key,value in unique_elements_cipher.items():
                count = len(value)
                unique_counts_cipher[key] = count
            unique_counts_plain = {}
            for key,value in unique_elements_plain.items():
                count = len(value)
                unique_counts_plain[key] = count

            result = {}
            for key1, value1 in unique_counts_plain.items():
                if functions.is_unique_value(unique_counts_plain, value1):
                    for key2, value2 in unique_counts_cipher.items():
                        if value2 == value1 and functions.is_unique_value(unique_counts_cipher, value2):
                            result[key2] = key1

            unique_frequency_cipher = functions.convert_nested_counts_to_frequencies(unique_elements_cipher)
            unique_frequency_plain = functions.convert_nested_counts_to_frequencies(unique_elements_plain)
            

            for key1, value1 in unique_counts_plain.items():
                # print(key1, value1)
                # print("=========")
                if key1 not in result.keys():
                    dict1 = unique_frequency_plain[key1]
                    # print(dict1)
                    for key2, value2 in unique_counts_cipher.items():
                        # print(key2, value2)
                        # print("---------")
                        if key2 not in result.values() and  abs(value1 - value2) < 365:
                            dict2 = unique_frequency_cipher[key2]
                            # print(dict2)
                            if dict2:
                                dis = functions.euclidean_distance(dict1, dict2)
                                # print(key1, key2, dis)
                                if dis < 0.1:
                                    result[key2] = key1
                                    break
            OPE_list = ['Age', 'Admission Type', 'Length of stay', 'Risk']
            DET_list = ['Discharge', 'Gender', 'Race']


            countOPE = 0
            countDET = 0
            record_count = len(matrix) - 1
            
            for key, value in result.items():
                if key == value and key in OPE_list:
                    countOPE += 1
                elif key == value and key in DET_list:
                    countDET += 1
            OPEdict[record_count] = countOPE
            DETdict[record_count] = countDET
            column_count[record_count] = len(unique_elements_plain)
            recovered_count[record_count] = countDET + countOPE

            
            line = f"{str(flag)}: {len(unique_elements_plain)} columns need recovery, {countDET + countOPE} columns successfully recovered, OPE: {countOPE}, DET: {countDET}\n"
            flag += 1
            FinalResult.append(line)
            
            
        outputPath = os.path.join(out, f"{str(il)}q2010_{str(f)}.txt")
        print(outputPath)

        with open(outputPath, 'w') as outputfile:
            for line in FinalResult:
                outputfile.write(line)
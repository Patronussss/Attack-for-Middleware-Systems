import sys # 引用上级目录中的文件
sys.path.append("F:/Desktop/Supplementary experiments") 
import functions
import random
import time
import os


def SortingAttack(filePath, filePathPlain):
    startTime = time.time()
    matrix_cipher = functions.read_csv_to_matrix(filePath)

    # 统计到底有多少个关键字
    selected_columns_ope_withoutid = ['Age', 'Admission Type', 'Length of stay', 'Risk']
    matrix_ope_withoutid = functions.generate_submatrix(matrix_cipher, selected_columns_ope_withoutid)
    keyword_count = functions.count_keywords(matrix_ope_withoutid[1:])

    # 生成密文矩阵
    selected_columns_ope = ['Record ID', 'Age', 'Admission Type', 'Length of stay', 'Risk']
    matrix_cipher_ope = functions.generate_submatrix(matrix_cipher, selected_columns_ope)

    # 生成明文矩阵
    matrix_plain = functions.read_csv_to_matrix(filePathPlain)
    matrix_plain_ope = functions.generate_submatrix(matrix_plain, selected_columns_ope)

    def custom_sort(row):
        return tuple(int(cell) if cell.isdigit() else cell for cell in row[1:])

    # 按照要求对矩阵进行排序
    sorted_matrix_cipher = sorted(matrix_cipher_ope[1:], key=custom_sort)
    sorted_matrix_plain = sorted(matrix_plain_ope[1:], key=custom_sort)

    value_mapping = {}

    for i in range(len(sorted_matrix_cipher)):
        row_cipher = sorted_matrix_cipher[i][1:]  # 去掉第一列
        row_plain = sorted_matrix_plain[i][1:]   # 去掉第一列
        if row_cipher == row_plain:
            # matching_pairs.append((sorted_matrix_cipher[i][0], sorted_matrix_plain[i][0]))  # 记录第一列的值
            for j in range(len(row_cipher)):
                value_mapping[row_cipher[j]] = row_plain[j]

    endTime = time.time()
    totalTime = round(endTime - startTime, 2)
    # print("耗时: {:.2f}秒".format(endTime - startTime))

    accuracy = 0
    count = 0
    for key, value in value_mapping.items():
        if key == value:
            count += 1

    accuracy = count / keyword_count
    return totalTime, accuracy


if __name__ == '__main__':
    root = "F:/Desktop/Attack for Datablinder/"
    quarters = ['1q2010', '2q2010', '3q2010', '4q2010']
    out = 'result/output of SortingAttack-ours.txt'
    # 打开输出文件
    with open(out, 'w', encoding='utf-8') as f:
        for base in quarters:
            rootPath = os.path.join(root,base)
            file_list = os.listdir(rootPath)
            for file_name in file_list:
                filePath = os.path.join(rootPath,file_name)
                filePathPlain = os.path.join(rootPath,file_name)
                name = os.path.join(base, file_name)
                print(name)
                
                # 每个文件运行10次取平均值
                total_times = []
                accuracies = []
                for _ in range(10):
                    totalTime, accuracy = SortingAttack(filePath, filePathPlain)
                    total_times.append(totalTime)
                    accuracies.append(accuracy)
                
                # 计算平均值
                avg_time = sum(total_times) / 10
                avg_accuracy = sum(accuracies) / 10
                
                # 将结果写入文件
                f.write(f"文件路径: {name}, 平均执行时间: {avg_time:.2f}秒, 平均准确率: {avg_accuracy:.4f}\n")
                f.write("-" * 50 + "\n")

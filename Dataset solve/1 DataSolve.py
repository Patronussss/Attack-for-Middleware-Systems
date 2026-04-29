import csv
import json
import os

# 从JSON文件中读取数据
def load_icd9_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

# 递归解析ICD-9-CM代码
def parse_icd9_codes(data, code_dict):
    for entry in data:
        code = entry['code']
        desc = entry['desc']
        code_dict[code] = desc
        if 'children' in entry and entry['children']:
            parse_icd9_codes(entry['children'], code_dict)

# 将ICD-9-CM代码转换为英文疾病名称
def get_disease_name(icd_code, code_dict):
    return code_dict.get(icd_code, "Unknown code")

# 避免出现科学计数法
def format_cell(cell):
    if isinstance(cell, (int, float)):
        return '{:.20f}'.format(cell).rstrip('0').rstrip('.')
    return str(cell)

base = 'E:\\code\\Searchable Encryption\\dataset\\PUDF\\ZLDS\\2019'
list = os.listdir(base)
for i in list:
    if i.endswith('.txt'):
        input_file = base + '/' + i
        output_file = base + '/' + i[:-4] + '.csv'

        with open(input_file, 'r', encoding='utf-8') as file:  
            lines = file.readlines()

        res = []
        res.append(['Discharge','Hospital', 'Gender', 'Admission Type', 'Length of stay',
                    'Age', 'Race', 'Pincipal Diagnosis', 'Risk',
                    'Record ID'])
        icd_9_cm_path = 'F:\\Desktop\\Supplementary experiments\\Dataset solve\\diagnosis_codes_ICD_10_CM.json'
        icd_9_data = load_icd9_json(icd_9_cm_path)

        icd_9_code = {}
        parse_icd9_codes(icd_9_data, icd_9_code)
        index = [[747, 55],[47, 1],[24, 1],[51, 4],
                 [55, 2],[48, 1],[143, 7],[716, 1],
                 [0, 12],[12,6]]
        for line in lines:
            Discharge = line[index[9][0]:index[9][0] + index[9][1]].strip()
            Hospital = line[index[0][0]:index[0][0] + index[0][1]].strip()
            if Hospital == 'Low Discharge Volume Hospital' or Hospital == ' ' or Hospital == 'unknown':
                continue
            Sex = line[index[1][0]:index[1][0] + index[1][1]]
            if Sex == 'U' or Sex == ' ':
                continue

            if line[index[2][0]:index[2][0] + index[2][1]] == '1':
                Type_of_Admission = 'Emergency'
            elif line[index[2][0]:index[2][0] + index[2][1]] == '2':
                Type_of_Admission = 'Urgent'
            elif line[index[2][0]:index[2][0] + index[2][1]] == '3':
                Type_of_Admission = 'Elective'
            elif line[index[2][0]:index[2][0] + index[2][1]] == '4':
                Type_of_Admission = 'Newborn'
            elif line[index[2][0]:index[2][0] + index[2][1]] == '5':
                Type_of_Admission = 'Trauma_Center'
            else:
                Type_of_Admission = 'Others'

            if line[index[3][0]:index[3][0] + index[3][1]] == '    ':
                continue
            Length_of_Stay = line[index[3][0]:index[3][0] + index[3][1]]

            age_mapping = {
                "00": "1-28_days",
                "01": "29-365_days",
                "02": "1-4_years",
                "03": "5-9_years",
                "04": "10-14_years",
                "05": "15-17_years",
                "06": "18-19_years",
                "07": "20-24_years",
                "08": "25-29_years",
                "09": "30-34_years",
                "10": "35-39_years",
                "11": "40-44_years",
                "12": "45-49_years",
                "13": "50-54_years",
                "14": "55-59_years",
                "15": "60-64_years",
                "16": "65-69_years",
                "17": "70-74_years",
                "18": "75-79_years",
                "19": "80-84_years",
                "20": "85-89_years",
                "21": "90+_years",
                "22": "0-17_years",
                "23": "18-44_years",
                "24": "45-64_years",
                "25": "65-74_years",
                "26": "75+_years",
                "*": "Invalid"
            }
            if line[index[4][0]:index[4][0] + index[4][1]] == "":
                continue
            Age = age_mapping.get(line[index[4][0]:index[4][0] + index[4][1]])

            race_mapping = {
                "1": "American_Indian/Eskimo/Aleut",
                "2": "Asian_or_Pacific_Islander",
                "3": "Black",
                "4": "White",
                "5": "Other",
                "*": "Invalid"
            }
            Race = race_mapping.get(line[index[5][0]:index[5][0] + index[5][1]])

            if line[49:50] == "1":
                Race = "Hispanic"
            if Race == "":
                continue

            Princ_Diag_Code_num = line[index[6][0]:index[6][0] + index[6][1]].strip()
            if len(Princ_Diag_Code_num) <= 3:
                Princ_Diag_Code_num = Princ_Diag_Code_num
            else:
                Princ_Diag_Code_num = Princ_Diag_Code_num[:3]  # 只分大类
            print(Princ_Diag_Code_num)

            Princ_Diag_Code = get_disease_name(Princ_Diag_Code_num, icd_9_code)
            Princ_Diag_Code = Princ_Diag_Code.replace(',', '-')

            if Princ_Diag_Code_num == '    ':
                continue

            risk_mapping = {
                "1": "Minor",
                "2": "Moderate",
                "3": "Major",
                "4": "Extreme" 
            }

            if line[index[7][0]:index[7][0] + index[7][1]] == '':
                continue
            risk = risk_mapping.get(line[index[7][0]:index[7][0] + index[7][1]])

            Record_id = format_cell(line[index[8][0]:index[8][0] + index[8][1]])

            res.append([Discharge, Hospital, Sex, Type_of_Admission, Length_of_Stay, Age, Race, 
                        Princ_Diag_Code, risk, 
                        Record_id])

        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)

            for row in res:
                csvwriter.writerow(row)



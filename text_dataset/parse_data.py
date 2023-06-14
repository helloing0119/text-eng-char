import os
import csv

csv_file = 'english.csv'


# CSV 파일에 저장할 데이터 리스트
data = []

def get_label(num_in_str):
    num = int(num_in_str)
    if num<=10:
        return str(num-1)
    if num<=36:
        return chr(65+num-11)
    return chr(97 + num -37)

for cnum in range(1, 63):
    for fnum in range(1, 56):
        data.append(["Img/img"+str(cnum).zfill(3) +"-" + str(fnum).zfill(3)+".png",
                     get_label(str(cnum))])
data = [["image","label"]]+data

# 데이터를 CSV 파일에 저장
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print("CSV 파일이 생성되었습니다.")
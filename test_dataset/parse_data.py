import os
import csv

csv_file = 'test_data.csv'

# 이미지를 찾을 폴더 경로
folder_path = './test_dataset'

# CSV 파일에 저장할 데이터 리스트
data = []

# 폴더 이름이 'SampleXXX'인 폴더들을 찾음
folders = [folder for folder in os.listdir(folder_path) if folder.startswith('Sample')]

def get_label(num_in_str):
    num = int(num_in_str)
    if num<=10:
        return str(num-1)
    if num<=36:
        return chr(65+num-11)
    return chr(97 + num -37)

# 폴더별로 이미지 파일들을 찾아서 데이터에 추가
for folder in folders:
    folder_full_path = os.path.join(folder_path, folder)
    label = get_label(folder[7:])
    if os.path.isdir(folder_full_path):
        images = [image for image in os.listdir(folder_full_path) if image.endswith(('.jpg', '.jpeg', '.png'))]
        for i, image in enumerate(images):
            if i>20:
                image_full_path = os.path.join(folder_full_path, image)
                data.append([folder+"/" +image, label])
    data = [["image","label"]]+sorted(data, key=lambda x: x[0])

# 데이터를 CSV 파일에 저장
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print("CSV 파일이 생성되었습니다.")
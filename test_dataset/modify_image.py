import os
import csv
import cv2
import random
import numpy as np


csv_file = 'test_data.csv'

# 이미지를 찾을 폴더 경로
folder_path = './test_dataset'

# CSV 파일에 저장할 데이터 리스트
data = []


def add_noise(image, noise_level):
    """
    이미지에 노이즈를 추가합니다.
    
    :param image: 입력 이미지 (NumPy 배열)
    :param noise_level: 노이즈 강도 (0.0 ~ 1.0)
    :return: 노이즈가 추가된 이미지 (NumPy 배열)
    """
    h, w, c = image.shape
    noisy_image = np.copy(image)
    
    # 노이즈 생성
    noise = np.random.randn(h, w, c) * 255 * noise_level
    
    # 노이즈 추가
    noisy_image = cv2.add(noisy_image, noise.astype(np.uint8))
    
    return noisy_image

def tilt_image(image, angle):
    """
    이미지를 지정된 각도로 기울입니다.
    
    :param image: 입력 이미지 (NumPy 배열)
    :param angle: 기울이기 각도 (시계 방향으로 -90 ~ 90도)
    :return: 기울여진 이미지 (NumPy 배열)
    """
    h, w, c = image.shape
    _angle = angle+360 if angle < 0 else angle
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, _angle, 1.0)
    tilted_image = cv2.warpAffine(image, matrix, (w, h))
    
    return tilted_image

def random_move(image):
    # 중심 위치 조정
    height, width, _ = image.shape
    center_x = width // 2
    center_y = height // 2
    
    shift_x = random.randint(int(-width/10), int(width/10))
    shift_y = random.randint(int(-height/10), int(height/10))
    
    new_center_x = center_x + shift_x
    new_center_y = center_y + shift_y
    
    translation_matrix = np.float32([[1, 0, new_center_x - center_x], [0, 1, new_center_y - center_y]])
    transformed_image = cv2.warpAffine(image, translation_matrix, (width, height))
    
    return transformed_image

folders = [folder for folder in os.listdir(folder_path) if folder.startswith('Sample')]

# 폴더별로 이미지 파일들을 찾아서 데이터에 추가
for folder in folders:
    folder_full_path = os.path.join(folder_path, folder)
    if os.path.isdir(folder_full_path):
        images = [image for image in os.listdir(folder_full_path) if image.endswith(('.jpg', '.jpeg', '.png'))]
        for i, image in enumerate(images):
            if i>20:
                image_full_path = os.path.join(folder_full_path, image)
                image = cv2.imread(image_full_path)
                # 이미지에 노이즈를 추가합니다.
                noisy_image = add_noise(image, noise_level=random.randint(0, 50)/10)

                # 이미지를 기울입니다.
                tilted_image = tilt_image(noisy_image, angle=random.randint(-20, 20))

                treansfered_image = random_move(tilted_image)
                cv2.imwrite(image_full_path, treansfered_image)
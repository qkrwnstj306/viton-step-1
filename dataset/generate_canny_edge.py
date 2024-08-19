import cv2 as cv
import os
import numpy as np

def process_images(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        # 이미지 파일인지 확인 (예: 확장자 체크)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 이미지 읽기
            image = cv.imread(input_path)
            gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            # 이미지가 제대로 읽혔는지 확인
            if gray_img is not None:
                # Canny Edge Detection 적용
                edges = cv.Canny(gray_img, 100, 200)
                
                # 출력 경로 설정
                output_path = os.path.join(output_dir, filename)
                
                # 결과 이미지 저장
                cv.imwrite(output_path, edges)
                print(f'Saved: {output_path}')
            else:
                print(f'Failed to load image: {input_path}')

if __name__ == "__main__":
    # 입력 디렉토리와 출력 디렉토리 설정
    dir_name = "test" # "test" or "train"

    input_directory = f'./zalando-hd-resized/{dir_name}/cloth'  # 입력 디렉토리 경로
    output_directory = f'./zalando-hd-resized/{dir_name}/cloth_canny'  # 출력 디렉토리 경로

    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 이미지 처리 실행
    process_images(input_directory, output_directory)
import os
import json

def create_json(folder_path, is_train):
    # JSON 파일에 저장할 데이터 리스트 생성
    data_list = []

    # 각 폴더의 경로 생성
    image_folder_path = os.path.join(folder_path, 'image')

    # 이미지 폴더 내의 파일 목록 가져오기
    image_files = os.listdir(image_folder_path)
    
    # 파일 목록을 기반으로 데이터 리스트 생성
    for file_name in image_files:
        data_list.append({
            'input_image': os.path.join("image", file_name),
            'cloth': os.path.join("cloth", file_name),
            'person_agnostic_mask': os.path.join("agnostic-mask",file_name[:-4] + "_mask.png"),
            'cloth_agnostic_mask': os.path.join("agnostic-v3.2", file_name),
            'densepose': os.path.join("image-densepose", file_name),
            'cloth_mask': os.path.join("cloth-mask", file_name)
        })

    # 결과를 JSON 파일로 저장
    if is_train:
        json_path = './metadata.json'
    else:
        json_path = './test_metadata.json'    
        
    with open(json_path, 'w') as json_file:
        for data in data_list:
            json.dump(data, json_file)
            json_file.write('\n')

    print(f"JSON 파일이 성공적으로 생성되었습니다. 경로: {json_path}")

IS_TRAIN = False #change this

# 실행할 폴더의 경로를 지정
if IS_TRAIN:
    folder_path = '../../../HuggingFace-ControlNet/ready-dataset/zalando-hd-resized/train'
else:
    folder_path = '../../../HuggingFace-ControlNet/ready-dataset/zalando-hd-resized/test'

# JSON 파일 생성 함수 호출
create_json(folder_path, IS_TRAIN)

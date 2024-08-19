import torch
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision import transforms
from PIL import Image
import os

def pil_to_tensor(img):
    """
    PIL 이미지를 torch.Tensor로 변환합니다.
    """
    img_tensor = transforms.functional.to_tensor(img)  # 이미지를 텐서로 변환
    img_tensor = img_tensor.byte()  # dtype을 torch.uint8로 변경
    return img_tensor

def cal_kid(original_dir, generated_dir):
    kid = KernelInceptionDistance(subset_size=50, normalize=True) # if image range is in [0,1], set normalize=True

    original_files = os.listdir(original_dir)
    generated_files = os.listdir(generated_dir)

    original_batch = []  # 실제 이미지 미니 배치를 저장하기 위한 리스트
    generated_batch = []  # 생성된 이미지 미니 배치를 저장하기 위한 리스트

    for file in original_files:
        if file.endswith(".jpg") or file.endswith(".png"):  # 이미지 파일인 경우에만 처리
            img_path = os.path.join(original_dir, file)
            img = Image.open(img_path)
            img = img.resize((384, 512))  # 모델 입력 크기에 맞게 조정
            img_tensor = pil_to_tensor(img)  # 이미지를 torch.Tensor로 변환 -> range: [0,1]
            original_batch.append(img_tensor)
            if len(original_batch) == 50:  # 미니 배치 크기에 도달하면
                # 미니 배치로 업데이트
                original_batch_tensor = torch.stack(original_batch)
                kid.update(original_batch_tensor, real=True)
                # 미니 배치 초기화
                original_batch = []

    for file in generated_files:
        if file.endswith(".jpg") or file.endswith(".png"):  # 이미지 파일인 경우에만 처리
            img_path = os.path.join(generated_dir, file)
            img = Image.open(img_path)
            img = img.resize((384, 512))  # 모델 입력 크기에 맞게 조정
            img_tensor = pil_to_tensor(img)  # 이미지를 torch.Tensor로 변환 -> range: [0,1]
            generated_batch.append(img_tensor)
            if len(generated_batch) == 50:  # 미니 배치 크기에 도달하면
                # 미니 배치로 업데이트
                generated_batch_tensor = torch.stack(generated_batch)
                kid.update(generated_batch_tensor, real=False)
                # 미니 배치 초기화
                generated_batch = []

    kid_mean, kid_std = kid.compute() 
    print(kid_mean, kid_std)

if __name__=='__main__':
    idx = 100
    generated_dir_lst = [f'../v2/outputs/epoch-{idx}/unpaired/cfg_5',
                         ]
    
    original_dir = '../dataset/zalando-hd-resized/test/image'
    generated_dir = generated_dir_lst[0]
    
    cal_kid(original_dir, generated_dir)

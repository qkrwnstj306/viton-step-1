import torch, os
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.strategies.ddp import DDPStrategy
import cv2
from dataset import MyTrainDataset, collate_fn
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from pytorch_lightning import seed_everything

import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(fmt='%(name)s: %(lineno)4d - %(levelname)s - %(message)s',level='INFO') # ['INFO', 'DEBUG']

# Custom

from args import arguments
import model_loader
from utils import check_gpu_memory_usage, tensor2img
from test_model import MyModel

# Hyperparameters 

WIDTH = 384     
HEIGHT = 512   
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

"""
CUDA_VISIBLE_DEVICES=2 python inference_all_ckpt.py --seed 23 --paired True --batch_size 4
"""

def setting(weight="epoch=0-step=0.ckpt"):
    # Load Arguments 

    args = arguments()
    seed_everything(args.seed)
    args.save_dir = "./outputs/" + weight.split("-")[0].replace("=", "-")

    # Load Parameter of SDv1.5
    logger.info("MODEL LOAD...!")
    ckpt_pth = f'./weights/eps/eps_23/eps/{weight}'
    models = model_loader.load_models_from_fine_tuned_weights(ckpt_pth, 'cpu')
    logger.info("MODEL LOAD COMPLETE...!")

    # Load Dataset 
    logger.info("Data LOAD...!")
    test_dataset = MyTrainDataset(args.debugging, is_train=False, is_paired=args.paired)
    
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, 
                                                   shuffle=False, pin_memory=True, num_workers=4)
        
    logger.info("Data LOAD COMPLETE...!")
    
    return args, models, test_dataloader

@torch.no_grad()
def main_worker(args, models, data_module):
    is_paired = "paired"
    if not args.paired:
        is_paired = "unpaired"
    save_dir = args.save_dir + f'/{is_paired}/cfg_{args.cfg_scale}'
    os.makedirs(save_dir, exist_ok=True)
    # Model Setting 
    diffusion = models["diffusion"]
    dinov2 = models["dinov2"]
    mlp = models["mlp"]     
    decoder = models["decoder"] 
    encoder = models["encoder"]
    state_dict = models["state_dict"]
    
    model = MyModel(encoder, decoder, diffusion, dinov2, mlp, args)
    model.load_state_dict(state_dict, strict=True)
    model = model.to("cuda")

    del models
    
    # Train and Validate 

    for batch_idx, batch in enumerate(data_module):
        print(f"{batch_idx}/{len(data_module)}")
        predicted_images, CFG = model.test_step(batch, batch_idx)
        
        
        for predicted_image, input_image, agn_mask, img_fn, cloth_fn in zip(predicted_images, batch['input_image'], 
                                                     batch['agn_mask'], batch['img_fn'], batch['cloth_fn']):
            
            # range: [0, 255], dim: [512, 384, 3], type: numpy
            # input_image range:[-1, 1], dim: [3, 512, 384], type: torch
            # agn_mask range:[0, 1], dim: [1, 512, 384], type: torch, inference 시에는 agn_mask 가 inversion 되어있지 않다.
            predicted_image = tensor2img(predicted_image)
            
            if args.blending:
                import numpy as np 
                input_image = np.uint8(255.0 * ((input_image+1.0) / 2.0).permute(1,2,0).cpu().numpy())
                agn_mask = agn_mask.permute(1,2,0).cpu().numpy()
                
                predicted_image = agn_mask * predicted_image + (1. - agn_mask) * input_image
            
            filename = f"{img_fn.split('.')[0]}_{cloth_fn.split('.')[0]}.jpg"
            save_pth = os.path.join(save_dir, filename)
            cv2.imwrite(save_pth, predicted_image[:,:,::-1])

if __name__ == "__main__":
    folder_path = './weights/eps/eps_23/eps'
    ckpt_files = [f for f in os.listdir(folder_path) if f.endswith('.ckpt')]
    
    # 파일 이름에 포함된 숫자가 5의 배수인지 확인하는 필터링 조건 추가
    import re
    ckpt_files = [
        ckpt for ckpt in ckpt_files
        if re.search(r'epoch=(\d+)-step=\d+\.ckpt', ckpt) and int(re.search(r'epoch=(\d+)-step=\d+\.ckpt', ckpt).group(1)) % 5 == 0
    ]
    print(ckpt_files)
    for ckpt in ckpt_files:
        args, models, data_module = setting(ckpt)
        is_paired = "paired"
        if not args.paired:
            is_paired = "unpaired"
        if os.path.exists(args.save_dir + f'/{is_paired}/cfg_{args.cfg_scale}') and os.path.isdir(args.save_dir + f'/{is_paired}/cfg_{args.cfg_scale}'):
            print(f"디렉터리 '{args.save_dir}:{is_paired}'가 존재합니다.")
            continue
        
        main_worker(args, models, data_module)

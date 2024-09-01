import torch, os
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.strategies.ddp import DDPStrategy
import cv2
from dataset import MyTrainDataset, collate_fn
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import make_grid, save_image

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

def setting():
    # Load Arguments 

    args = arguments()

    # Load Parameter of SDv1.5
    logger.info("MODEL LOAD...!")
    ckpt_pth = './weights/eps/epoch=65-step=15834.ckpt'
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
    args, models, data_module = setting()
    main_worker(args, models, data_module)

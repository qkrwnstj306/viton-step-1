import torch, os
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.strategies.ddp import DDPStrategy
from dataset import MyTrainDataset, collate_fn
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import cv2
from pytorch_lightning import seed_everything

import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(fmt='%(name)s: %(lineno)4d - %(levelname)s - %(message)s',level='INFO') # ['INFO', 'DEBUG']

# Custom

from args import arguments
import model_loader
from utils import check_gpu_memory_usage, tensor2img
from hook import CrossAttentionHook
from test_model import MyModel

# Hyperparameters 

WIDTH = 384     
HEIGHT = 512   
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

"""
CUDA_VISIBLE_DEVICES=1 python attention_inference.py --generated_image False --specific_reference_attribution_maps True --patch_index {6, 7, 8} --save_dir ./outputs/epoch-0 --only_one_data True --certain_data_idx 00865_00.jpg --seed 23
"""

def setting():
    # Load Arguments 

    args = arguments()
    seed_everything(args.seed)
    
    # Load Parameter of SDv1.5
    logger.info("MODEL LOAD...!")
   
    ckpt_pth = './weights/eps/attn_loss/epoch=149-step=46228.ckpt'
    models = model_loader.load_models_from_fine_tuned_weights(ckpt_pth, 'cpu')
    logger.info("MODEL LOAD COMPLETE...!")

    # Load Dataset 
    logger.info("Data LOAD...!")
    test_dataset = MyTrainDataset(args.debugging, is_train=False, is_paired=args.paired, 
                                  only_one_data=args.only_one_data, certain_data_idx=args.certain_data_idx)
    
    # Must be 1 for batch_size
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, 
                                                   shuffle=False, pin_memory=True, num_workers=4)
        
    logger.info("Data LOAD COMPLETE...!")
    
    return args, models, test_dataloader

@torch.no_grad()
def main_worker(args, models, data_module):
    is_paired = "paired"
    if not args.paired:
        is_paired = "unpaired"
    save_dir = args.save_dir + f'/{is_paired}/attention/cfg_{args.cfg_scale}'
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
    
    hook = CrossAttentionHook(args)
    hook.take_module(model)

    del models
    
    # Train and Validate 
    IMACS = 0
    
    for batch_idx, batch in enumerate(data_module):
        print(f"{batch_idx}/{len(data_module)}")
        predicted_image, CFG = model.test_step(batch, batch_idx)
        # predicted_image: [1, 3, 512, 384]
        # input_image range:[-1, 1], dim: [1, 3, 512, 384], type: torch
        # agn_mask range:[0, 1], dim: [1, 1, 512, 384], type: torch, inference 시에는 agn_mask 가 inversion 되어있지 않다.
        # cloth_mask range: [0, 1], dim: [1, 1, 512, 384]
        
        
        attention_maps = hook.make_attention_maps(batch['agn_mask'])
        clamp_predicted_image = torch.clamp(predicted_image, min=-1.0, max=1.0)
        
        if hook.specific_reference_attribution_maps:
            for idx in range(len(attention_maps)):
                hook.make_specific_reference_attribution_maps(clamp_predicted_image, batch['cloth'], attention_maps[idx], save_dir, f"{batch_idx}-{idx}", batch['img_fn'][0].split('.')[0])
        
        elif hook.cal_IMACS:
            IMACS += hook.make_images(clamp_predicted_image, batch['cloth'], attention_maps, save_dir, f"{batch_idx}", batch['agn_mask'], batch['cloth_mask'])
            print(IMACS)
        else:
            if hook.per_head:
                for idx, attention_map in enumerate(attention_maps,1):
                    hook.make_images(clamp_predicted_image, batch['cloth'], attention_map, save_dir, f"{batch_idx}-head-{idx}")
            elif hook.per_time: # TO DO
                for idx, key in enumerate(attention_maps.keys(),1):
                    hook.make_images(clamp_predicted_image, batch['cloth'], attention_maps[key], save_dir, f"{batch_idx}-time-{idx}")
            elif not hook.per_head and not hook.per_time:
                hook.make_images(clamp_predicted_image, batch['cloth'], attention_maps, save_dir, batch_idx)

        filename = f"{batch['img_fn'][0].split('.')[0]}_{batch['cloth_fn'][0].split('.')[0]}.jpg"
        save_pth = os.path.join(save_dir, filename)
        predicted_image = tensor2img(predicted_image)
        cv2.imwrite(save_pth, predicted_image[:,:,::-1])
    
    print(f"TOTAL SCORES: {IMACS/len(data_module)}")
    
if __name__ == "__main__":
    args, models, data_module = setting()
    main_worker(args, models, data_module)

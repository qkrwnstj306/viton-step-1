import torch 
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.strategies.ddp import DDPStrategy

import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(fmt='%(name)s: %(lineno)4d - %(levelname)s - %(message)s',level='INFO') # ['INFO', 'DEBUG']

# Custom

from args import arguments
from model import LiTModel
import model_loader
from callback import model_checkpoint_setting, lr_monitor_setting, SaveImageCallback
from pl_dataset import VITONDataModule
from utils import check_gpu_memory_usage

# Hyperparameters 

WIDTH = 384     
HEIGHT = 512   
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def setting():
    # Load Arguments 

    args = arguments()

    # Load Parameter of SDv1.5
    logger.info("STATE DICT LOAD...!")
    model_file = './v2/weights/eps/epoch=95-step=34944.ckpt'
    models = model_loader.preload_models_from_standard_weights(model_file, 'cpu')
    logger.info("STATE DICT LOAD COMPLETE...!")

    # Load Dataset 
    logger.info("Data LOAD...!")
    data_module = VITONDataModule(args)
    logger.info("Data LOAD COMPLETE...!")
    
    return args, models, data_module

def main_worker(args, models, data_module):

    # Model Setting 

    diffusion = models["diffusion"]
    dinov2 = models["dinov2"]
    mlp = models["mlp"]     
    decoder = models["decoder"] 
    encoder = models["encoder"]
    state_dict = models["state_dict"]
    
    # Train and Validate 
    model = LiTModel(diffusion, mlp, dinov2, decoder, encoder, args)
    model.load_state_dict(state_dict, strict=False)
    model.diffusion.copy_diffusion_weights_to_controlnet()
    del models

    tb_logger = loggers.TensorBoardLogger('./logs')
    
    checkpoint_callback = model_checkpoint_setting()
    save_image_callback = SaveImageCallback(batch_frequency=args.batch_frequency)
    lr_monitor = lr_monitor_setting()

    if args.resume:
        resume_from_checkpoint = './weights/eps/epoch=20-step=7644.ckpt'
    else:
        resume_from_checkpoint = None
        
    trainer = pl.Trainer(gpus=args.n_gpus,
                        resume_from_checkpoint=resume_from_checkpoint,
                        max_epochs=args.n_epochs,
                        callbacks=[checkpoint_callback, save_image_callback, lr_monitor],
                        precision=16,
                        accelerator='gpu',
                        accumulate_grad_batches=args.accumulation_steps,
                        strategy='ddp',
                        logger=tb_logger,
                        )

    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    args, models, data_module = setting()
    main_worker(args, models, data_module)

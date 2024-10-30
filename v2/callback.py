import os
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision.utils import make_grid, save_image
from args import arguments

args = arguments()

def lr_monitor_setting():
    return LearningRateMonitor(logging_interval='epoch')

# def model_checkpoint_setting():

#     return ModelCheckpoint(
#         dirpath=f'./weights/{args.weight_dir}/',
#         filename='{epoch}-{step}',
#         save_top_k=-1,
#         every_n_epochs=5,
#         verbose=True,
#         save_on_train_epoch_end=True
#     )

class CustomCheckpoint(ModelCheckpoint):
    def on_train_epoch_end(self, trainer, pl_module):
        # 현재 에포크가 5의 배수일 때만 체크포인트를 저장
        if (trainer.current_epoch) % 5 == 0:  # +1은 0-based 인덱스 처리
            super().on_train_epoch_end(trainer, pl_module)

def model_checkpoint_setting():
    return CustomCheckpoint(
        dirpath=f'./weights/{args.weight_dir}/',
        filename='{epoch}-{step}',
        save_top_k=-1,
        verbose=True,
    )

class SaveImageCallback(Callback):
    def __init__(self, batch_frequency=1,save_dir=f"./image_log"):
        super().__init__()

        self.batch_freq = batch_frequency
        self.save_dir = save_dir

    @rank_zero_only
    def log_local(self, input_image, predicted_images, cloth_agnostic_mask, densepose, cloth, global_step, current_epoch, batch_idx, CFG):
        if not os.path.isdir(self.save_dir):
              os.makedirs(self.save_dir)
        filename = "gs-{:06}_e-{:06}_b-{:06}_cfg-{}.png".format(global_step, current_epoch, batch_idx,CFG)
        save_path = f"{self.save_dir}/{filename}"

        grid_images = torch.cat((input_image, predicted_images, cloth_agnostic_mask, densepose, cloth), dim=0)
        grid = make_grid(grid_images, normalize=True)
        save_image(grid, save_path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if self.check_frequency(check_idx):

          is_train = pl_module.training
          if is_train:
              pl_module.eval()

          with torch.no_grad():
              input_image, predicted_images, cloth_agnostic_mask, densepose, cloth, CFG = pl_module.log_images(batch, batch_idx)

          self.log_local(input_image, predicted_images, cloth_agnostic_mask, densepose, cloth, pl_module.global_step, pl_module.current_epoch, batch_idx, CFG)

          if is_train:
              pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        self.log_img(pl_module, batch, batch_idx, split="train")
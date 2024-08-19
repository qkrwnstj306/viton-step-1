import pytorch_lightning as pl
from dataset import MyTrainDataset, collate_fn
import torch, gc
from utils import check_gpu_memory_usage

class VITONDataModule(pl.LightningDataModule):
    
    def __init__(self, args):
        super().__init__()

        self.args = args
        
    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        self.train_dataset = MyTrainDataset(self.args.debugging, is_train=True)
        
    def train_dataloader(self):
        
        batch_size = max(self.args.batch_size//self.args.n_gpus, 1)
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, collate_fn=collate_fn, 
                                                   shuffle=True, pin_memory=True, num_workers=4)
        
        return train_dataloader
        
    def val_dataloader(self):
        pass
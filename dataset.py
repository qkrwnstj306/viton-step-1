import json, cv2
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import albumentations as A
from einops import rearrange
from os.path import join

def imread(pth, w=384, h=512, is_mask=False):
    
    img = cv2.imread(pth)
    
    if is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (w,h))
        img = (img >= 128).astype(np.float32)  # 0 or 1
        img = img[:,:,None]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w,h))
        # Normalize images to [-1, 1].
        img = (img.astype(np.float32) / 127.5) - 1.0
        
    return img

def imread_for_albu(
        p, 
        is_mask=False, 
        cloth_mask_check=False,
):
    img = cv2.imread(p)
    
    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = (img>=128).astype(np.float32)
        if cloth_mask_check:
            if img.sum() < 30720*4:
                img = np.ones_like(img).astype(np.float32)
        # range: 0 or 255
        img = np.uint8(img*255.0)
    return img

def norm_for_albu(img, is_mask=False):
    if not is_mask:
        # range: [-1, 1]
        img = (img.astype(np.float32)/127.5) - 1.0
    else:
        # range: 0 or 1
        img = img.astype(np.float32) / 255.0
        img = img[:,:,None]
    return img

class MyTrainDataset(Dataset):
    def __init__(self, debugging=False, is_train=True, is_paired=True, only_one_data=False, certain_data_idx=None):
        
        self.is_train = is_train 
        self.debugging = debugging
        
        self.only_one_data = only_one_data
        self.certain_data_idx = certain_data_idx
        
        self.pair_key = "paired" if is_paired else "unpaired"
        self.data_dir = './dataset/zalando-hd-resized'
        
        self.size_transform = A.Compose([A.Resize(512, 384)],
                                        additional_targets={
                                            "cloth":"image",
                                            "cloth_mask":"image",
                                            "agn":"image",
                                            "densepose":"image",
                                            "agn_mask":"image",
                                            "warped_cloth_mask":"image",
                                            "canny":"image"
                                        }
                                        )
        
        self.spatial_transform = A.Compose([
            A.HorizontalFlip(p=0.5), 
        ],
                additional_targets={
                                    "cloth": "image", 
                                    "cloth_mask": "image",
                                    "agn":"image",  
                                    "densepose":"image",
                                    "agn_mask":"image",
                                    "warped_cloth_mask":"image",
                                    "canny":"image"
                                    }
        )
        
        self.cloth_transform = A.Compose([ 
                A.ShiftScaleRotate(rotate_limit=0, shift_limit=0.2, scale_limit=(-0.2, 0.2), border_mode=cv2.BORDER_CONSTANT, p=0.5, value=0), 
            ],
                additional_targets={
                                    "cloth_mask":"image",
                                    "canny":"image"
                                    }               
            )
        
        self.person_transform = A.Compose([
                A.ShiftScaleRotate(rotate_limit=0, shift_limit=0.2, scale_limit=(-0.2, 0.2), border_mode=cv2.BORDER_CONSTANT, p=0.5, value=0), 
            ],
                additional_targets={
                                    "agn":"image",  
                                    "densepose":"image",
                                    "agn_mask":"image",
                                    "warped_cloth_mask":"image",
                                    }
            )
        
        self.color_transform = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.02), contrast_limit=(-0.3, 0.3), p=0.5), # all
                A.HueSaturationValue(5,5,5,p=0.5), # all
            ],
                additional_targets={
                                    "agn":"image",  
                                    "cloth":"image",
                                    }
            )
        
        if self.is_train:
            self.data_pth = "train"
            
        else: 
            self.data_pth = "test"
        
        im_names = []
        c_names = []
        
        with open(join(self.data_dir, f"{self.data_pth}_pairs.txt"), "r") as f:
            for index, line in enumerate(f.readlines(),1):
                im_name, c_name = line.strip().split()    
                if self.only_one_data:
                    if self.certain_data_idx == im_name:
                        im_names.append(im_name)
                        c_names.append(c_name)
                        break
                else:
                    im_names.append(im_name)
                    c_names.append(c_name)
                    if self.debugging:
                        if index == 5:
                            break    
                
        self.im_names = im_names
        self.c_names = dict()
        self.c_names["paired"] = im_names
        self.c_names["unpaired"] = c_names

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, idx):
        
        img_fn = self.im_names[idx]
        cloth_fn = self.c_names[self.pair_key][idx]

        if self.is_train:
            input_image = imread_for_albu(join(self.data_dir, self.data_pth, "image", self.im_names[idx]))
            cloth = imread_for_albu(join(self.data_dir, self.data_pth, "cloth", self.c_names[self.pair_key][idx]))
            cloth_agnostic_mask = imread_for_albu(join(self.data_dir, self.data_pth, "agnostic-v3.2", self.im_names[idx]))
            cloth_mask = imread_for_albu(join(self.data_dir, self.data_pth, "cloth-mask", self.c_names[self.pair_key][idx]), is_mask=True)
            densepose = imread_for_albu(join(self.data_dir, self.data_pth, "image-densepose", self.im_names[idx]))
            agn_mask = imread_for_albu(join(self.data_dir, self.data_pth, "agnostic-mask", self.im_names[idx].replace('.jpg', "_mask.png")), is_mask=True, cloth_mask_check=True)
            warped_cloth_mask = imread_for_albu(join(self.data_dir, self.data_pth, "gt_cloth_warped_mask", self.im_names[idx]), is_mask=True)
            canny = imread_for_albu(join(self.data_dir, self.data_pth, "cloth_canny", self.im_names[idx]), is_mask=True)
            
            
            transformed_size = self.size_transform(
                image=input_image,
                cloth=cloth,
                agn=cloth_agnostic_mask,
                cloth_mask=cloth_mask,
                densepose=densepose,
                agn_mask=agn_mask,
                warped_cloth_mask=warped_cloth_mask,
                canny=canny
            )
            
            input_image = transformed_size['image']
            cloth = transformed_size['cloth']
            cloth_agnostic_mask = transformed_size['agn']
            cloth_mask = transformed_size['cloth_mask']
            densepose = transformed_size['densepose']
            agn_mask = transformed_size['agn_mask']
            warped_cloth_mask = transformed_size['warped_cloth_mask']
            canny = transformed_size['canny']
            
            transformed_spatial = self.spatial_transform(
                image=input_image,
                cloth=cloth,
                cloth_mask=cloth_mask,
                agn=cloth_agnostic_mask,
                densepose=densepose,
                agn_mask=agn_mask,
                warped_cloth_mask=warped_cloth_mask,
                canny=canny
            )
            
            input_image = transformed_spatial["image"]
            cloth =transformed_spatial["cloth"]
            cloth_mask = transformed_spatial["cloth_mask"]
            cloth_agnostic_mask = transformed_spatial["agn"]
            densepose = transformed_spatial["densepose"]
            agn_mask = transformed_spatial["agn_mask"]
            warped_cloth_mask = transformed_spatial["warped_cloth_mask"]
            canny = transformed_spatial["canny"]
            
            transformed_cloth = self.cloth_transform(image=cloth,
                                         cloth_mask=cloth_mask,
                                         canny=canny)
            
            cloth = transformed_cloth["image"]
            cloth_mask = transformed_cloth["cloth_mask"]
            canny = transformed_cloth["canny"]
            
            transformed_person = self.person_transform(
                image=input_image,
                agn=cloth_agnostic_mask,
                densepose=densepose,
                agn_mask=agn_mask,
                warped_cloth_mask=warped_cloth_mask,
            )
            
            input_image = transformed_person["image"]
            cloth_agnostic_mask = transformed_person["agn"]
            densepose = transformed_person["densepose"]
            agn_mask = transformed_person["agn_mask"]
            warped_cloth_mask = transformed_person["warped_cloth_mask"]

            transformed_color = self.color_transform(
                image=input_image,
                agn=cloth_agnostic_mask,
                cloth=cloth
            )
            
            input_image = transformed_color["image"]
            cloth_agnostic_mask = transformed_color["agn"]
            cloth = transformed_color["cloth"]
            
            agn_mask = 255 - agn_mask
            cloth_agnostic_mask = cloth_agnostic_mask * agn_mask[:,:,None].astype(np.float32)/255.0 + 128 * (1 - agn_mask[:,:,None].astype(np.float32)/255.0) 
            
            input_image = norm_for_albu(input_image)
            cloth =norm_for_albu(cloth)
            cloth_mask = norm_for_albu(cloth_mask, is_mask=True)
            cloth_agnostic_mask = norm_for_albu(cloth_agnostic_mask)
            densepose = norm_for_albu(densepose)
            agn_mask = norm_for_albu(agn_mask, is_mask=True)
            agn_mask = 1 - agn_mask
            warped_cloth_mask = norm_for_albu(warped_cloth_mask, is_mask=True)
            canny = norm_for_albu(canny, is_mask=True)
            
        else:
            # normalized images
            input_image = imread(join(self.data_dir, self.data_pth, "image", self.im_names[idx]))
            cloth = imread(join(self.data_dir, self.data_pth, "cloth", self.c_names[self.pair_key][idx]))
            cloth_agnostic_mask = imread(join(self.data_dir, self.data_pth, "agnostic-v3.2", self.im_names[idx]))
            cloth_mask = imread(join(self.data_dir, self.data_pth, "cloth-mask", self.c_names[self.pair_key][idx]), is_mask=True)
            densepose = imread(join(self.data_dir, self.data_pth, "image-densepose", self.im_names[idx]))
            agn_mask = imread(join(self.data_dir, self.data_pth, "agnostic-mask", self.im_names[idx].replace('.jpg', "_mask.png")), is_mask=True)
            canny = imread(join(self.data_dir, self.data_pth, "cloth_canny", self.c_names[self.pair_key][idx]), is_mask=True)
            warped_cloth_mask = np.zeros_like(agn_mask)
            
            transformed_size = self.size_transform(
                image=input_image,
                cloth=cloth,
                agn=cloth_agnostic_mask,
                cloth_mask=cloth_mask,
                densepose=densepose,
                agn_mask=agn_mask,
                warped_cloth_mask=warped_cloth_mask,
                canny=canny
            )
            
            input_image = transformed_size['image']
            cloth = transformed_size['cloth']
            cloth_agnostic_mask = transformed_size['agn']
            cloth_mask = transformed_size['cloth_mask']
            densepose = transformed_size['densepose']
            agn_mask = transformed_size['agn_mask']
            warped_cloth_mask = transformed_size['warped_cloth_mask']
            canny = transformed_size['canny']
            
        return dict(input_image=input_image, cloth=cloth, cloth_agnostic_mask=cloth_agnostic_mask,
                    cloth_mask=cloth_mask, densepose=densepose, agn_mask=agn_mask, warped_cloth_mask=warped_cloth_mask,
                    img_fn=img_fn, cloth_fn=cloth_fn, canny=canny)
    
def collate_fn(samples):
    input_image = [torch.from_numpy(sample["input_image"]) for sample in samples]
    cloth = [torch.from_numpy(sample["cloth"]) for sample in samples]
    cloth_agnostic_mask = [torch.from_numpy(sample["cloth_agnostic_mask"]) for sample in samples]
    cloth_mask = [torch.from_numpy(sample['cloth_mask']) for sample in samples]
    densepose = [torch.from_numpy(sample["densepose"]) for sample in samples]
    agn_mask = [torch.from_numpy(sample["agn_mask"]) for sample in samples]
    warped_cloth_mask = [torch.from_numpy(sample["warped_cloth_mask"]) for sample in samples]
    canny = [torch.from_numpy(sample["canny"]) for sample in samples]
    img_fn = [sample["img_fn"] for sample in samples]
    cloth_fn = [sample["cloth_fn"] for sample in samples]
    
    input_image = torch.stack(input_image)
    cloth = torch.stack(cloth)
    cloth_agnostic_mask = torch.stack(cloth_agnostic_mask)
    cloth_mask = torch.stack(cloth_mask)
    densepose = torch.stack(densepose)
    agn_mask = torch.stack(agn_mask)
    warped_cloth_mask = torch.stack(warped_cloth_mask)
    canny = torch.stack(canny)
    
    input_image = rearrange(input_image, 'b h w c -> b c h w')
    cloth = rearrange(cloth, 'b h w c -> b c h w')
    cloth_agnostic_mask = rearrange(cloth_agnostic_mask, 'b h w c -> b c h w')
    cloth_mask = rearrange(cloth_mask, 'b h w c -> b c h w')
    densepose = rearrange(densepose, 'b h w c -> b c h w')
    agn_mask = rearrange(agn_mask, 'b h w c -> b c h w')
    warped_cloth_mask = rearrange(warped_cloth_mask, 'b h w c -> b c h w')
    canny = rearrange(canny, 'b h w c -> b c h w')
    
    batch = {
        'input_image': input_image, 
        'cloth': cloth,
        'cloth_agnostic_mask': cloth_agnostic_mask,
        'cloth_mask': cloth_mask,
        'densepose': densepose,
        'agn_mask': agn_mask,
        'warped_cloth_mask': warped_cloth_mask,
        'canny': canny,
        'img_fn': img_fn,
        'cloth_fn': cloth_fn
    }
    
    return batch


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = MyTrainDataset(is_train=True)
    dataloader = DataLoader(dataset, num_workers=0, collate_fn=collate_fn, batch_size=2, shuffle=True)
    
    batch = next(iter(dataloader))
    input_image = batch['input_image']
    cloth = batch['cloth']
    cloth_agnostic_mask = batch['cloth_agnostic_mask']
    cloth_mask = batch['cloth_mask']
    densepose = batch['densepose']
    agn_mask = batch['agn_mask']
    warped_cloth_mask = batch['warped_cloth_mask']
    canny = batch['canny']
    
    from torchvision.utils import make_grid, save_image
    
    grid_images = torch.cat((input_image, cloth, cloth_agnostic_mask, densepose), dim=0)
    grid = make_grid(grid_images, normalize=True)
    save_image(grid, './dataset_test_grid.png')
    
    save_image(cloth_mask, './dataset_test_cloth_mask.png')
    
    save_image(agn_mask, './dataset_test_agn_mask.png')
    
    save_image(warped_cloth_mask, './dataset_test_warped_cloth_mask.png')
    save_image(canny, './dataset_test_canny.png')
    #breakpoint()
    
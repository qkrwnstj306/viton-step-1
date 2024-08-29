import torch
import torch.nn as nn
from attention import CrossAttention, SelfAttention
from collections import defaultdict
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
import os 
from matplotlib.patches import Rectangle

class CrossAttentionHook(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.generated_image = args.generated_image
        self.per_head = args.per_head
        self.per_time = args.per_time
        
        self.specific_reference_attribution_maps = args.specific_reference_attribution_maps
        self.cloth_pixel_visualization = args.cloth_pixel_visualization
        self.height = 0
        self.width = 0
        self.patch_index = str(args.patch_index) #[0,14]
        
        self.cal_IMACS = args.cal_IMACS
        self.cloth_forcing = args.cloth_forcing
        
        self.current_time_step = 0
        self.number_of_group = 5
        self.layer_num = 15
        if self.per_time:
            self.cross_attention_forward_hooks = defaultdict(lambda: defaultdict(int))
        else:
            self.cross_attention_forward_hooks = defaultdict(lambda:0)
        self.cross_attention_forward_hooks_for_loss = defaultdict(lambda:0)

    def clear(self):
        self.cross_attention_forward_hooks.clear()
        self.cross_attention_forward_hooks_for_loss.clear()
        self.current_time_step = 0
        self.height = 0
        self.width = 0 
    
    def cal_IMACS_score(self, attention_maps, agn_mask, cloth_mask):
        # resized_attention_maps
        # range: [0,1], size: [512, 384], type: numpy
        # resized_attention_maps[resized_attention_maps >= 0.6] = 1, 즉, 값들이 현재 plot 을 위해 반전되어있다.
        attention_maps = 1. - attention_maps 
    
        # agn_mask range:[0, 1], dim: [1, 1, 512, 384], type: torch, inference 시에는 agn_mask 가 inversion 되어있지 않다.
        # cloth_mask range: [0, 1], dim: [1, 1, 512, 384]
        if self.generated_image:
            agn_mask = agn_mask.squeeze(0).squeeze(0).cpu().numpy()
            inversion_agn_mask = 1. - agn_mask
            
            masked_region_score = (np.sum(attention_maps * agn_mask)) / np.sum(agn_mask)    
            non_masked_region_score = (np.sum(attention_maps * inversion_agn_mask)) / np.sum(inversion_agn_mask)

        else:
            cloth_mask = cloth_mask.squeeze(0).squeeze(0).cpu().numpy()
            inversion_cloth_mask = 1. - cloth_mask
            
            masked_region_score = (np.sum(attention_maps * cloth_mask)) / (np.sum(cloth_mask) + 1e-8)
            non_masked_region_score = (np.sum(attention_maps * inversion_cloth_mask)) / (np.sum(inversion_cloth_mask) + 1e-8)
        
        penalty = 3.
        IMACS = masked_region_score - penalty * non_masked_region_score
        
        return IMACS 
    
    def make_specific_reference_attribution_maps(self, input_image, cloth, attention_maps, save_dir, batch_idx):
        with torch.cuda.amp.autocast(dtype=torch.float32):
            # input_image: [1, 3, 512, 384]
            # cloth_image: [1, 3, 512, 384]
            # attention_maps: [64, 48]   
            attention_maps, idx = attention_maps[0], attention_maps[1]
            # range: [0, 1]
            attention_maps = (attention_maps - attention_maps.min()) / (attention_maps.max() - attention_maps.min() + 1e-8)
            
            # [1, 3, 512, 384] -> [1, 32, 24, 3] -> [32, 24, 3]
            input_image = F.interpolate(input_image, size=(self.height, self.width), mode='bicubic').clamp_(min=0).permute(0,2,3,1)
            input_image = input_image[0]
            
            cloth_image = cloth[0]
            if self.cloth_pixel_visualization:
                resized_attention_maps = F.interpolate(attention_maps.cpu().unsqueeze(0).unsqueeze(0), size=(512,384), mode='bicubic')
                resized_attention_maps = ((resized_attention_maps - resized_attention_maps.min()) / (resized_attention_maps.max() - resized_attention_maps.min() + 1e-8)).squeeze(0).squeeze(0).numpy()
            else:
                cloth_image = F.interpolate(cloth_image.unsqueeze(0), size=(37,28), mode='bicubic').squeeze(0)
                resized_attention_maps = attention_maps.cpu().squeeze(0).squeeze(0).numpy()
            
            input_image = np.uint8(((input_image + 1.0) * 127.5).numpy())
            cloth_image = np.uint8(((cloth_image.permute(1,2,0) + 1.0) * 127.5).numpy())

            resized_attention_maps = 1.0 - resized_attention_maps
            resized_attention_maps[resized_attention_maps >= 0.6] = 1 
            resized_attention_maps = cv2.applyColorMap(np.uint8(resized_attention_maps*255), cv2.COLORMAP_JET)

            heat_map = cv2.addWeighted(cloth_image, 0.7, resized_attention_maps, 0.5, 0)
            
            attention_map_dir = f"{save_dir}/specific"
            os.makedirs(attention_map_dir, exist_ok=True)
            
            plt.imshow(input_image)
            plt.gca().add_patch(Rectangle((idx[1]-0.5, 
                                           idx[0]-0.5), 1, 1, edgecolor='red'
                                          , facecolor='none'))  # 선택한 위치 주변에 빨간색 테두리 네모 추가
            plt.axis('off')
            attention_map_filename = f"idx-{batch_idx}_reference_attribution_map_generated_image.png"
            attetion_map_save_pth = f"{attention_map_dir}/{attention_map_filename}"
            
            plt.savefig(attetion_map_save_pth, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            plt.imshow(heat_map)
            heat_map_filename = f"idx-{batch_idx}_reference_attribution_map_cloth.png"
            heat_map_save_pth = f"{attention_map_dir}/{heat_map_filename}"
            plt.axis('off')
            plt.savefig(heat_map_save_pth, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            
    def make_images(self, input_image, cloth, attention_maps, save_dir, batch_idx, agn_mask=None, cloth_mask=None):
        with torch.cuda.amp.autocast(dtype=torch.float32):
            # input_image: [1, 3, 512, 384]
            # attention_maps: [64, 48]
            if self.cloth_forcing:
                assert self.generated_image == False, "Please set self.generated_image to False and self.cal_IMACS to True"
                # [32, 24]
                force_cloth_mask = F.interpolate(cloth_mask, size=(64,48), mode='bicubic').squeeze(0).squeeze(0)
                attention_maps = attention_maps.cpu() * force_cloth_mask.cpu()
            # range: [0, 1]
            attention_maps = (attention_maps - attention_maps.min()) / (attention_maps.max() - attention_maps.min() + 1e-8)
            
            if self.generated_image:
                input_image = input_image[0]
            else:
                input_image = cloth[0]
            # [3, 512, 384], range: [-1, 1] -> [512, 384, 3], range: [0, 255], type: numpy
            input_image = np.uint8(((input_image.permute(1,2,0) + 1.0) * 127.5).numpy())
            
            # [512, 384, 1] 
            resized_attention_maps = F.interpolate(attention_maps.cpu().unsqueeze(0).unsqueeze(0), size=(512,384), mode='bicubic')
            resized_attention_maps = ((resized_attention_maps - resized_attention_maps.min()) / (resized_attention_maps.max() - resized_attention_maps.min() + 1e-8)).squeeze(0).squeeze(0).numpy()
            
            resized_attention_maps = 1.0 - resized_attention_maps
            resized_attention_maps[resized_attention_maps >= 0.6] = 1 
            
            if self.cal_IMACS:
                IMACS = self.cal_IMACS_score(resized_attention_maps, agn_mask, cloth_mask)
                return IMACS
            
            resized_attention_maps = cv2.applyColorMap(np.uint8(resized_attention_maps*255), cv2.COLORMAP_JET)
            
            heat_map = cv2.addWeighted(input_image, 0.7, resized_attention_maps, 0.5, 0)
            
            attention_maps = attention_maps.cpu().numpy()
            attention_maps[attention_maps <= 0.4] = 0
            
            plt.imshow(attention_maps, cmap='jet')
            attention_map_filename = f"idx-{batch_idx}_attention_map_generated_image-{self.generated_image}.png"
            attetion_map_save_pth = f"{save_dir}/{attention_map_filename}"
            plt.axis('off')
            plt.savefig(attetion_map_save_pth, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            plt.imshow(heat_map)
            heat_map_filename = f"idx-{batch_idx}_heatmap_generated_image-{self.generated_image}.png"
            heat_map_save_pth = f"{save_dir}/{heat_map_filename}"
            plt.axis('off')
            plt.savefig(heat_map_save_pth, bbox_inches='tight', pad_inches=0)
            plt.close()
    
    def make_attention_maps(self, agn_mask=None):
        # 하나로 합치기 
        with torch.cuda.amp.autocast(dtype=torch.float32):
            # will be [8, seq_len, height, width] * 15
            if self.per_time:
                attention_maps = {}
                for time_group in range(self.number_of_group):
                    attention_maps[time_group] = []
            else:
                attention_maps = []
            
            if self.specific_reference_attribution_maps:
                        key_list = {
                            '0': ['encoders.1.1.attention_2', 64], # Encoders
                            '1': ['encoders.2.1.attention_2', 64],
                            '2': ['encoders.4.1.attention_2', 32],
                            '3': ['encoders.5.1.attention_2', 32],
                            '4': ['encoders.7.1.attention_2', 16],
                            '5': ['encoders.8.1.attention_2', 16],
                            '6': ['decoders.3.1.attention_2', 16],   # Decoders
                            '7': ['decoders.4.1.attention_2', 16],
                            '8': ['decoders.5.1.attention_2', 16],
                            '9': ['decoders.6.1.attention_2', 32],
                            '10': ['decoders.7.1.attention_2', 32],
                            '11': ['decoders.8.1.attention_2', 32],
                            '12': ['decoders.9.1.attention_2', 64],
                            '13': ['decoders.10.1.attention_2', 64],
                            '14': ['decoders.11.1.attention_2', 64],
                        }
                        # User Setting (str)
                        IDX = self.patch_index
                        key, height_cond = key_list[IDX]
                        
                        seq_len = self.cross_attention_forward_hooks[key].size(1)
                        self.width = int(math.sqrt((seq_len * 3 / 4)))
                        self.height = int(self.width * 4 / 3)
                        if self.height == height_cond:  # 37 28
                            # UNet 에서의 query(generated image) 가 32 resolution 인 attetion layer 에 대해서 specific-reference attribuiton map visualization
                            attention_maps.append(F.interpolate(self.cross_attention_forward_hooks[key], size=(37, 28), mode='bicubic').clamp_(min=0))
        
            else:
                for key in self.cross_attention_forward_hooks.keys():
                    if self.per_time:
                        for time_group in range(self.number_of_group):
                            attention_maps[time_group].append(F.interpolate(self.cross_attention_forward_hooks[key][time_group], size=(64, 48), mode='bicubic').clamp_(min=0))
                    else:
                        attention_maps.append(F.interpolate(self.cross_attention_forward_hooks[key], size=(64, 48), mode='bicubic').clamp_(min=0))

            if self.specific_reference_attribution_maps:
                assert not self.per_head and not self.per_time, "You have to set per_head and per_time False"
                #agn_mask range:[0, 1], dim: [1, 1, 512, 384], type: torch,

                # [1, 1, 512, 384] -> [1, 1, (32 x 24)]
                agn_mask = F.interpolate(agn_mask, size=(self.height,self.width), 
                                                  mode='nearest').flatten(start_dim=2,end_dim=3)
                # [1, 1, (32 x 24)] -> [(32 x 24)]
                agn_mask = agn_mask.squeeze(0).squeeze(0).cuda()
                grid = torch.ones(agn_mask.size()).cuda()
                 # ((32 x 24))
                true_region = (agn_mask * grid) != 0
                
                reference_attetion_maps = []
                for idx in range(true_region.size(0)):
                    if true_region[idx]:
                        selected_idx = (idx // self.width, idx % self.width) 
                        # [8, 1, 64, 48] 
                        reference_attention_map = [attention_map[:,idx,:,:] for attention_map in attention_maps]
                        reference_attention_map = torch.cat(reference_attention_map, dim=0).mean(0)
                        # [64, 48] x Number of True
                        reference_attetion_maps.append((reference_attention_map, selected_idx))
                
                return reference_attetion_maps
                
                
                
            if self.per_head:
                if self.generated_image:
                    for i, maps in enumerate(attention_maps):
                        attention_maps[i] = maps.unsqueeze(1)
                    attention_maps = torch.cat(attention_maps, dim=1)
                    attention_maps = list(torch.chunk(attention_maps, 8, dim=0))
                    
                    for i, maps in enumerate(attention_maps):
                        attention_maps[i] = maps.squeeze(0).mean(0).mean(0)
                   
                else: 
                    if agn_mask is not None:
                    # agn_mask range:[0, 1], dim: [1, 1, 512, 384], type: torch, inference 시에는 agn_mask 가 inversion 되어있지 않다.
                                 
                        for idx, attention_map in enumerate(attention_maps):
                            seq_len = attention_map.size(1)
                            width = int(math.sqrt((seq_len * 3 / 4)))
                            height = int(width * 4 / 3)
                            # [1, 1 , 512, 384] -> [1, 1, height, width] -> [1, 1, (height x width = seq_len)] -> [seq_len, 1, 1]
                            resized_agn_mask = F.interpolate(agn_mask, size=(height, width), mode='nearest').flatten(start_dim=2,end_dim=3).permute(2,1,0).cuda()
                            # [8, seq_len, 64, 48]
                            attention_maps[idx] = attention_map * resized_agn_mask
                    attention_maps = torch.cat(attention_maps, dim=1)
                    attention_maps = list(torch.chunk(attention_maps, 8, dim=0))
                    for i, maps in enumerate(attention_maps):
                        attention_maps[i] = maps.squeeze(0).mean(0)
                        
            elif self.per_time:
                if self.generated_image:
                    for time_group in range(self.number_of_group):
                        attention_maps[time_group] = torch.cat(attention_maps[time_group], dim=0)
                        attention_maps[time_group] = attention_maps[time_group].mean(0).mean(0)
                else:
                    for time_group in range(self.number_of_group): 
                        if agn_mask is not None:
                    # agn_mask range:[0, 1], dim: [1, 1, 512, 384], type: torch, inference 시에는 agn_mask 가 inversion 되어있지 않다.
                                 
                            for idx, attention_map in enumerate(attention_maps[time_group]):
                                seq_len = attention_map.size(1)
                                width = int(math.sqrt((seq_len * 3 / 4)))
                                height = int(width * 4 / 3)
                                # [1, 1 , 512, 384] -> [1, 1, height, width] -> [1, 1, (height x width = seq_len)] -> [seq_len, 1, 1]
                                resized_agn_mask = F.interpolate(agn_mask, size=(height, width), mode='nearest').flatten(start_dim=2,end_dim=3).permute(2,1,0).cuda()
                                # [8, seq_len, 64, 48]
                                attention_maps[time_group][idx] = attention_map * resized_agn_mask
                        attention_maps[time_group] = [am.mean(dim=1) for am in attention_maps[time_group]]
                        attention_maps[time_group] = torch.cat(attention_maps[time_group], dim=0)
                        attention_maps[time_group] = attention_maps[time_group].mean(0)
                        
            elif not self.per_head and not self.per_time: # time-head attribution maps
                # [8, seq_len, height, width] * 15 -> ... -> [64, 48]
                if self.generated_image:
                    # [120, seq_len, 64, 48]
                    attention_maps = torch.cat(attention_maps, dim=0)
                    attention_maps = attention_maps.mean(0).mean(0)
                else:
                    if agn_mask is not None:
                    # agn_mask range:[0, 1], dim: [1, 1, 512, 384], type: torch, inference 시에는 agn_mask 가 inversion 되어있지 않다.
                                 
                        for idx, attention_map in enumerate(attention_maps):
                            seq_len = attention_map.size(1)
                            width = int(math.sqrt((seq_len * 3 / 4)))
                            height = int(width * 4 / 3)
                            # [1, 1 , 512, 384] -> [1, 1, height, width] -> [1, 1, (height x width = seq_len)] -> [seq_len, 1, 1]
                            resized_agn_mask = F.interpolate(agn_mask, size=(height, width), mode='nearest').flatten(start_dim=2,end_dim=3).permute(2,1,0).cuda()
                            # [8, seq_len, 64, 48]
                            attention_maps[idx] = attention_map * resized_agn_mask
                    attention_maps = [attention_map.mean(dim=1) for attention_map in attention_maps]
                    attention_maps = torch.cat(attention_maps, dim=0)
                    attention_maps = attention_maps.mean(0)
            self.clear()
        
        return attention_maps
    
    def cross_attention_hook(self, module, input, output, name):
        # Get heat maps
        x, y = input[0], input[1]

        # [num_heads, context_seq_len, height, width]
        attention_score = module.get_attention_score(x, y, generated_image=self.generated_image)

        if self.per_time:
            if self.current_time_step < 10:
                group_name = 0
            elif 10 <= self.current_time_step < 20:
                group_name = 1
            elif 20 <= self.current_time_step < 30:
                group_name = 2
            elif 30 <= self.current_time_step < 40:
                group_name = 3
            elif 40 <= self.current_time_step < 50:
                group_name = 4
            else:
                raise ValueError
            if type(self.cross_attention_forward_hooks[name]) == int:
                self.cross_attention_forward_hooks[name][group_name] = attention_score
            else:
                self.cross_attention_forward_hooks[name][group_name] += attention_score
                
            self.layer_num -= 1
            if self.layer_num == 0:
                self.current_time_step += 1
                self.layer_num = 15
                
        else:
            if type(self.cross_attention_forward_hooks[name]) == int:
                self.cross_attention_forward_hooks[name] = attention_score
            else:
                self.cross_attention_forward_hooks[name] += attention_score
            
    def take_module(self, model):
        for name, module in model.diffusion.unet.named_modules():
            if isinstance(module, CrossAttention) and not 'bottleneck' in name:
                module.register_forward_hook(lambda m, inp, out, n=name: self.cross_attention_hook(m, inp, out, n))

import torch
import torch.nn as nn
from torchvision import transforms
import os
from torch.nn import functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from ddim import DDIMSampler

WIDTH = 384     
HEIGHT = 512   
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

class MyModel(nn.Module):
    
    def __init__(self, encoder ,decoder, diffusion, dinov2, mlp, args):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.diffusion = diffusion
        self.dinov2 = dinov2
        self.mlp = mlp
        
        self.args = args
        self.latent_shape = (self.args.batch_size, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
        
        self.encoder.eval()
        self.decoder.eval()
        self.diffusion.eval()
        self.dinov2.eval()
        self.mlp.eval()
        
        self.generator = torch.Generator()
        self.generator.manual_seed(self.args.seed)
        self.sampler = DDIMSampler(self.generator, cfg_scale=self.args.cfg_scale, parameterization=self.args.parameterization, rescale=self.args.rescale)
        
        self.temperal_image_transforms = transforms.Compose([
        transforms.Resize((518,392), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        ])
    
    @torch.no_grad()
    def mask_resize(self, m, h, w, inverse=False):
        m = F.interpolate(m, (h,w), mode="nearest")
        if inverse:
            m = 1-m
        return m
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        input_image = batch['input_image'].to('cpu') # [batch_size, 3, 512, 384]  
        cloth_agnostic_mask = batch['cloth_agnostic_mask'].to('cuda') # [batch_size, 3, 512, 384]
        densepose = batch['densepose'].to('cuda') # [batch_size, 3, 512, 384] 
        cloth = batch['cloth'].to('cuda') # [batch_size, 3, 512, 384]
        cloth_mask = batch['cloth_mask'].to('cuda') # [batch_size, 1, 512, 384] > for Attention Loss 
        warped_cloth_mask = batch['warped_cloth_mask'].to('cuda') # [batch_size, 1, 512, 384] for Attention Loss
        agn_mask = batch['agn_mask'].to('cuda')
        resized_agn_mask = self.mask_resize(agn_mask, LATENTS_HEIGHT, LATENTS_WIDTH, inverse=False) # [BS, 1, 64, 48]
        canny = batch['canny'].to('cuda') # [batch_size, 1, 512, 384]
        
        cloth_for_image_encoder = self.temperal_image_transforms(cloth)
        
        if self.args.conv_hint:
                encoder_inputs = torch.cat((input_image, cloth_agnostic_mask, densepose, cloth), dim=0)
                
        else:
            encoder_inputs = torch.cat((input_image, cloth_agnostic_mask, densepose, cloth, canny), dim=0)
            
        encoder_noise = torch.randn((encoder_inputs.size(0), 4, LATENTS_HEIGHT, LATENTS_WIDTH), generator=self.generator)
        
        if self.args.conv_hint:
            input_latents, cloth_agnostic_mask_latents, densepose_latents, cloth_latents= \
                torch.chunk(self.encoder(encoder_inputs, encoder_noise.to("cuda")), 4, dim=0)
        else:
            input_latents, cloth_agnostic_mask_latents, densepose_latents, cloth_latents, canny = \
                torch.chunk(self.encoder(encoder_inputs, encoder_noise.to("cuda")), 5, dim=0)
            
        image_embeddings = self.dinov2(cloth_for_image_encoder)
        image_embeddings = self.mlp(image_embeddings)
        
        x_T = torch.randn(self.latent_shape, generator=self.generator).to("cuda")
        
        CFG = True if self.args.do_cfg else False
        
        x_0 = self.sampler.DDIM_sampling(self.diffusion, x_T, cloth_agnostic_mask_latents,
                                         densepose_latents, cloth_latents, resized_agn_mask, canny, image_embeddings, input_latents, do_cfg=CFG)
        
        predicted_images = self.decoder(x_0).to("cpu")
        cloth_agnostic_mask, densepose, cloth = cloth_agnostic_mask.to('cpu'), densepose.to('cpu'), cloth.to('cpu')
        
        return predicted_images, CFG
    
    def extract_feature_map(self, batch, batch_idx, block_index):
        input_image = batch['input_image'].to('cpu') # [batch_size, 3, 512, 384]  
        cloth_agnostic_mask = batch['cloth_agnostic_mask'].to('cuda') # [batch_size, 3, 512, 384]
        densepose = batch['densepose'].to('cuda') # [batch_size, 3, 512, 384] 
        cloth = batch['cloth'].to('cuda') # [batch_size, 3, 512, 384]
        agn_mask = batch['agn_mask'].to('cuda')
        resized_agn_mask = self.mask_resize(agn_mask, LATENTS_HEIGHT, LATENTS_WIDTH, inverse=False) # [BS, 1, 64, 48]
        canny = batch['canny'].to('cuda')
        
        unet_model = self.diffusion.unet
        save_interval = 50
        save_feature_dir = './outputs/feature_map'
        os.makedirs(save_feature_dir, exist_ok=True)
        
        self.sampler.tau = 1
        
        def ddim_sampler_callback(t):
            save_feature_map_callback(t)
        
        def save_feature_map_callback(t):
            save_feature_maps(unet_model.decoders, t)
        
        def save_feature_map(feature_map, filename):
            save_path = os.path.join(save_feature_dir, f"{filename}.pt")
            torch.save(feature_map, save_path)
            
        def save_feature_maps(blocks, t):
            block_idx = 0
            from tqdm import tqdm
            for block in tqdm(blocks, desc="Saving decoder blocks feature maps"):
                if "ResidualBlock" in str(type(block[0])):
                    if block_idx == block_index or block_index == -1:
                        save_feature_map(block[0].in_layers_features, f"batch-{batch_idx}_{block_idx}_in_layers_features_time_{t}")
                        save_feature_map(block[0].out_layers_features, f"batch-{batch_idx}_{block_idx}_out_layers_features_time_{t}")
                if len(block) > 1 and "AttentionBlock" in str(type(block[1])):
                    save_feature_map(block[1].attention_1.k, f"batch-{batch_idx}_{block_idx}_self_attn_k_time_{t}")
                    save_feature_map(block[1].attention_1.q, f"batch-{batch_idx}_{block_idx}_self_attn_q_time_{t}")
                    save_feature_map(block[1].attention_1.v, f"batch-{batch_idx}_{block_idx}_self_attn_v_time_{t}")

                    save_feature_map(block[1].attention_2.k, f"batch-{batch_idx}_{block_idx}_cross_attn_k_time_{t}")
                    save_feature_map(block[1].attention_2.q, f"batch-{batch_idx}_{block_idx}_cross_attn_q_time_{t}")
                    save_feature_map(block[1].attention_2.v, f"batch-{batch_idx}_{block_idx}_cross_attn_v_time_{t}")
                block_idx += 1
        
        cloth_for_image_encoder = self.temperal_image_transforms(cloth)
        
        encoder_inputs = torch.cat((cloth_agnostic_mask, densepose, cloth), dim=0)
        
        encoder_noise = torch.randn((encoder_inputs.size(0), 4, LATENTS_HEIGHT, LATENTS_WIDTH), generator=self.generator).to("cuda")
        
        cloth_agnostic_mask_latents, densepose_latents, cloth_latents = \
            torch.chunk(self.encoder(encoder_inputs, encoder_noise), 3, dim=0)
            
        image_embeddings = self.dinov2(cloth_for_image_encoder)
        image_embeddings = self.mlp(image_embeddings)
        
        x_T = torch.randn(self.latent_shape, generator=self.generator).to("cuda")
        
        CFG = True if self.args.do_cfg else False
        
        x_0 = self.sampler.DDIM_sampling(self.diffusion, x_T, cloth_agnostic_mask_latents,
                                         densepose_latents, cloth_latents, resized_agn_mask, canny, image_embeddings, do_cfg=CFG, img_callback=ddim_sampler_callback, callback_interval=save_interval)
        
        predicted_images = self.decoder(x_0).to("cpu")
        cloth_agnostic_mask, densepose, cloth = cloth_agnostic_mask.to('cpu'), densepose.to('cpu'), cloth.to('cpu')
        
        return predicted_images, CFG
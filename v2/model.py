import torch, gc
import pytorch_lightning as pl
import itertools
import numpy as np
from torchvision import transforms
from torch.nn import functional as F

# Custom

from attention import CrossAttention
from ddim import DDIMSampler
from utils import check_gpu_memory_usage
from lr_scheduler import CosineAnnealingWarmUpRestarts
from hook import CrossAttentionHook

WIDTH = 384     
HEIGHT = 512   
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

'''
GPU 용량이 부족하다면 model.py대신 model_lowvram.py를 사용하면 된다. 
from model_lowvram import LiTModel

+ model.py에 del, torch.cuda.empty_cache(), gc.collect() 함수가 사용되는데 이는 학교 서버로 돌려도 cuda out of memory가 나서 사용한다.
'''


class LiTModel(pl.LightningModule):

    def __init__(self, diffusion, mlp, dinov2, decoder, encoder,
                 args):
        super().__init__()
        
        self.diffusion = diffusion
        self.mlp = mlp
        self.dinov2 = dinov2
        self.decoder = decoder
        self.encoder = encoder
        
        self.generator = torch.Generator()
        self.args = args
        self.latent_shape = (max(args.batch_size//args.n_gpus, 1), 4, LATENTS_HEIGHT, LATENTS_WIDTH)
        self.sampler = DDIMSampler(self.generator, cfg_scale=self.args.cfg_scale, parameterization=self.args.parameterization, rescale=self.args.rescale)
        
        self.dinov2.requires_grad_(False)
        self.dinov2.eval()
        self.decoder.requires_grad_(False)
        self.decoder.eval()
        self.encoder.requires_grad_(False)
        self.encoder.eval()
    
    @torch.no_grad()
    def mask_resize(self, m, h, w, inverse=False):
        m = F.interpolate(m, (h,w), mode="nearest")
        if inverse:
            m = 1-m
        return m
    
    def training_step(self, batch, batch_idx):
        # print([param for param in model.parameters() if param.requires_grad])

        with torch.no_grad():
            
            input_image = batch['input_image'] # [batch_size, 3, 512, 384]  
            cloth_agnostic_mask = batch['cloth_agnostic_mask'] # [batch_size, 3, 512, 384]
            densepose = batch['densepose'] # [batch_size, 3, 512, 384] 
            cloth = batch['cloth'] # [batch_size, 3, 512, 384] > Image Encoder > Path-level
            cloth_mask = batch['cloth_mask'] # [batch_size, 1, 512, 384] > for Attention Loss 
            warped_cloth_mask = batch['warped_cloth_mask'] # [batch_size, 1, 512, 384] for Attention Loss
            agn_mask = batch['agn_mask']
            resized_agn_mask = self.mask_resize(agn_mask, LATENTS_HEIGHT, LATENTS_WIDTH, inverse=False) # [BS, 1, 64, 48]
            
            temperal_image_transforms = transforms.Compose([
            transforms.Resize((518,392), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            ])

            # Image Encoder Input, [batch_size, 3, 518, 392]
            cloth_for_image_encoder = temperal_image_transforms(cloth)
            
            encoder_inputs = torch.cat((input_image, cloth_agnostic_mask, densepose, cloth), dim=0)
            
            encoder_noise = torch.randn((encoder_inputs.size(0), 4, LATENTS_HEIGHT, LATENTS_WIDTH))

            input_image_latents, cloth_agnostic_mask_latents, densepose_latents, cloth_latents = \
                torch.chunk(self.encoder(encoder_inputs, encoder_noise.to("cuda")), 4, dim=0)

            # [batch_size * 2, 3, 518, 392] -> [batch_size * 2, 1037, 1536]
            image_embeddings = self.dinov2(cloth_for_image_encoder)

        del cloth_agnostic_mask, densepose, cloth,  
        cloth_for_image_encoder, encoder_inputs, encoder_noise
        torch.cuda.empty_cache()
        gc.collect()
        
        # [batch_size * 2, 1037, 1536] -> [batch_size, 1037, 768], [batch_size, 1037, 768]
        image_embeddings = self.mlp(image_embeddings)
        
        if not self.args.use_pixel_loss:
            is_decoder = None
        else: 
            is_decoder = self.decoder
        if self.args.use_attention_loss:
            naive_loss, cwg_loss, tv_loss, dcml_loss, pixel_tv_loss, pixel_l2_loss = \
                self.sampler.train(self.diffusion, input_image_latents, cloth_agnostic_mask_latents,   
                             densepose_latents, cloth_latents, resized_agn_mask, image_embeddings, warped_cloth_mask, self.args.do_cfg, 
                             use_attention_loss=self.args.use_attention_loss, decoder=is_decoder, input_image=input_image)
            
            self.log('cwg', cwg_loss, on_step=True, prog_bar=True, logger=True)
            self.log('tv', tv_loss, on_step=True, prog_bar=True, logger=True)
            self.log('dcml', dcml_loss, on_step=True, prog_bar=True, logger=True)
            self.log('pixel_tv', pixel_tv_loss, on_step=True, prog_bar=True, logger=True)
            self.log('pixel_l2', pixel_l2_loss, on_step=True, prog_bar=True, logger=True)
            self.log('naive', naive_loss, on_step=True, prog_bar=True, logger=True)
            loss = naive_loss + cwg_loss + tv_loss + dcml_loss + pixel_tv_loss + pixel_l2_loss
            
        else:
            naive_loss, pixel_tv_loss, pixel_l2_loss = self.sampler.train(self.diffusion, input_image_latents, cloth_agnostic_mask_latents, 
                                      densepose_latents, cloth_latents, resized_agn_mask, image_embeddings, warped_cloth_mask=None, do_cfg=self.args.do_cfg, 
                                      use_attention_loss=False, decoder=is_decoder, input_image=input_image)

            self.log('pixel_tv', pixel_tv_loss, on_step=True, prog_bar=True, logger=True)
            self.log('pixel_l2', pixel_l2_loss, on_step=True, prog_bar=True, logger=True)
            self.log('naive', naive_loss, on_step=True, prog_bar=True, logger=True)
            loss = naive_loss + pixel_tv_loss + pixel_l2_loss 

        del image_embeddings, input_image_latents, cloth_agnostic_mask_latents, densepose_latents, cloth_latents
        torch.cuda.empty_cache()
        gc.collect()
        
        return loss
        
    def configure_optimizers(self):
        params = list(self.mlp.parameters())
 
        if self.args.do_cfg:
            params.append(self.diffusion.uncond_vector)
        
        if self.args.sd_locked:
            params += list(self.diffusion.unet.encoders[0].parameters())
            params += list(self.diffusion.unet.decoders.parameters())
            params += list(self.diffusion.final.parameters())
            
            for name, module in self.diffusion.unet.encoders.named_modules():
                if isinstance(module, CrossAttention):
                    print(f"Found CrossAttention module in {name}")
                    params += list(module.parameters()) 
        
        else:
            params += list(self.diffusion.unet.parameters())
            params += list(self.diffusion.final.parameters())
        # params = list(self.diffusion.unet.encoders[0].parameters())
        # params += list(self.mlp.parameters())
        
        # if self.args.do_cfg:
        #     params.append(self.diffusion.uncond_vector)
        
        # if not self.args.sd_locked:
        #     params += list(self.diffusion.unet.decoders.parameters())
        #     params += list(self.diffusion.final.parameters())
            
        #     for name, module in self.diffusion.unet.encoders.named_modules():
        #         if isinstance(module, CrossAttention):
        #             print(f"Found CrossAttention module in {name}")
        #             params += list(module.parameters()) 
        
        # else:
        #     for name, module in self.diffusion.unet.named_modules():
        #         if isinstance(module, CrossAttention):
        #             print(f"Found CrossAttention module in {name}")
        #             params += list(module.parameters()) 
        
        optimizer = torch.optim.AdamW(
                    params,
                    lr=self.args.lr if not self.args.scheduler else 0,
                    betas=(0.9, 0.999),
                    weight_decay=1e-2,
                    eps=1e-08,
                    )
        
        if self.args.scheduler:
            scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=6, T_mult=1, eta_max=0.00003,  T_up=2, gamma=0.7, last_epoch=-1)
            return [optimizer], [scheduler]
        
        return optimizer
    
    def log_images(self, batch, batch_idx):
        

        
        input_image = batch['input_image'].to('cpu') # [batch_size, 3, 512, 384]  
        cloth_agnostic_mask = batch['cloth_agnostic_mask'] # [batch_size, 3, 512, 384]
        densepose = batch['densepose'] # [batch_size, 3, 512, 384] 
        cloth = batch['cloth'] # [batch_size, 3, 512, 384]
        agn_mask = batch['agn_mask'] # [BS, 1, 512, 384]
        resized_agn_mask = self.mask_resize(agn_mask, LATENTS_HEIGHT, LATENTS_WIDTH, inverse=False) # [BS, 1, 64, 48]
        
        temperal_image_transforms = transforms.Compose([
        transforms.Resize((518,392), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        ])

        # Image Encoder Input, [batch_size, 3, 518, 392], [batch_size, 3, 518, 392]
        cloth_for_image_encoder = temperal_image_transforms(cloth)
        
        encoder_inputs = torch.cat((cloth_agnostic_mask, densepose, cloth), dim=0)
        
        encoder_noise = torch.randn((encoder_inputs.size(0), 4, LATENTS_HEIGHT, LATENTS_WIDTH))
        
        cloth_agnostic_mask_latents, densepose_latents, cloth_latents = \
            torch.chunk(self.encoder(encoder_inputs, encoder_noise.to("cuda")), 3, dim=0)

        # [batch_size * 2, 3, 518, 392] -> [batch_size * 2, 1037, 1536]
        image_embeddings = self.dinov2(cloth_for_image_encoder)
        
        del cloth_for_image_encoder, encoder_noise, encoder_inputs
        torch.cuda.empty_cache()
        gc.collect()
        
        # [batch_size * 2, 1037, 1536] -> [batch_size, 1037, 768], [batch_size, 1037, 768]
        image_embeddings = self.mlp(image_embeddings)
                        
        x_T = torch.randn(self.latent_shape).to("cuda")
        
        CFG = True if np.random.rand(1)[0] < 0.5 and self.args.do_cfg else False # Can generates either cfg/None
    
        x_0 = self.sampler.DDIM_sampling(self.diffusion, x_T, cloth_agnostic_mask_latents,
                                         densepose_latents, cloth_latents, resized_agn_mask, image_embeddings, do_cfg=CFG)
        
        predicted_images = self.decoder(x_0).to("cpu")
        
        cloth_agnostic_mask, densepose, cloth = cloth_agnostic_mask.to('cpu'), densepose.to('cpu'), cloth.to('cpu')
        
        del x_0, x_T, image_embeddings, resized_agn_mask
        torch.cuda.empty_cache()
        gc.collect()
        return input_image, predicted_images, cloth_agnostic_mask, densepose, cloth, CFG
        
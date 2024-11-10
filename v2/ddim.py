import torch, gc
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn 

import logging

logger = logging.getLogger(__name__)

class DDIMSampler:

    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.00085, beta_end: float = 0.0120, cfg_scale: float = 5.0, parameterization: str = "v", rescale: float = 0.5):
        self.generator = generator
        self.cfg_scale = cfg_scale
        self.parameterization = parameterization
        assert self.parameterization in ["eps", "v"]
        self.rescale = rescale
        
        self.T = num_training_steps
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        if self.parameterization == "v":
            self.betas = self.enforce_zero_terminal_snr(self.betas)
        
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_prev = F.pad(self.alphas_bar[:-1],(1,0), value=1.)
        self.sigma = self.betas*((1. - self.alpha_bar_prev)/(1. - self.alphas_bar))
        
        #for training
        self.sqrt_alpha_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1-self.alphas_bar)

        #for DDIM sampling
        self.scheduling = 'uniform'
        self.tau = 20 # uniform - time step 을 몇 번 건너 뛸 건지 / exp - time step 을 몇 번 거쳐서 생성할 건지

        self.eta = 0 
        
        self.pixel_loss = PixelLoss()
    
    def enforce_zero_terminal_snr(self, betas):
        # Convert betas to alphas_bar_sqrt
        alphas = 1 - betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

        # Shift so the last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T

        # Scale so the first timestep is back to the old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        # Convert alphas_bar_sqrt to betas
        alphas_bar = alphas_bar_sqrt ** 2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
        betas = 1 - alphas

        return betas
    
    def gather_and_expand(self, coeff, t, xshape):
        # [1000] / [batch_size] / [batch_size, 4, 64, 48]

        # batch_size / (4, 64, 48)
        batch_size, *dims = xshape
        
        # T 개의 coeff 중에서, index가 t인 것들을 추출
        coeff_t = torch.gather(coeff, dim=0, index=t)
        
        # coeff_t를 각 batch에 곱할 수 있도록 reshape, 각 pixel마다 같은 값을 coefficient로 곱해주기 때문에 뒤의 차원이 (1,1,1)
        return coeff_t.view(batch_size, 1, 1, 1)
    
    def get_v(self, x, noise, t):
        return self.gather_and_expand(self.sqrt_alpha_bar.to("cuda"), t.to("cuda"), x.shape) * noise.to("cuda") - self.gather_and_expand(self.sqrt_one_minus_alpha_bar.to("cuda"), t.to("cuda"), x.shape) * x.to("cuda")
    
    def predict_eps_from_z_and_v(self, x_t, t, v, batch_size):
        return self.sqrt_alpha_bar[t].to("cuda").view(batch_size, 1, 1, 1) * v + torch.sqrt(1 - self.alphas_bar[t].to("cuda")).view(batch_size, 1, 1, 1) * x_t
    
    def predict_start_from_z_and_v(self, x_t, t, v, batch_size):
        return self.sqrt_alpha_bar[t].to("cuda").view(batch_size, 1, 1, 1) * x_t - torch.sqrt(1 - self.alphas_bar[t].to("cuda")).view(batch_size, 1, 1, 1) * v
    
    def train(self, model, x_0, cloth_agnostic_mask,
              densepose, cloth, resized_agn_mask, image_embeddings, warped_cloth_mask, do_cfg,
              use_attention_loss=False, decoder=None, input_image=None):
        # x_T: [batch_size, 4, Height / 8, Width / 8]
        # person_agnostic_mask, densepose, cloth, cloth_mask: [batch_size, 4, Height / 8, Width / 8]
        # cloth_embeddings, person_embeddings: [batch_size, 1037, 768]
        
        # t: [batch_size]
        t = torch.randint(self.T, size=(x_0.shape[0],))

        for index, time in enumerate(t):
            if index == 0:
                temp = self.get_time_embedding(time)
            else:
                temp = torch.cat([temp, self.get_time_embedding(time)], dim = 0)
        
        # [batch_size, 160 * 2]
        time_embedding = temp
        
        # [batch_size, 64, 48]
        eps = torch.randn_like(x_0)
        
        # [batch_size, 4, 64, 48]
        x_t = self.gather_and_expand(self.sqrt_alpha_bar.to("cuda"), t.to("cuda"), x_0.shape) * x_0 \
        + self.gather_and_expand(self.sqrt_one_minus_alpha_bar.to("cuda"), t.to("cuda"), x_0.shape) * eps.to("cuda")
        
        if use_attention_loss:
            predicted_eps, cwg_loss, tv_loss, dcml_loss = \
                model(x_t, cloth_agnostic_mask, densepose, cloth, resized_agn_mask, image_embeddings, time_embedding.to("cuda"),
                                                    warped_cloth_mask=warped_cloth_mask, do_cfg=do_cfg, 
                                                    use_attention_loss=use_attention_loss)
            
            if self.parameterization == "eps":
                target = eps
            elif self.parameterization == "v":
                target = self.get_v(x_0, eps, t)
            
            naive_loss = F.mse_loss(predicted_eps, target.to("cuda"))
            
        else:
            predicted_eps = model(x_t, cloth_agnostic_mask, densepose, cloth, resized_agn_mask, image_embeddings, time_embedding.to("cuda"), do_cfg=do_cfg, use_attention_loss=False)
            if self.parameterization == "eps":
                target = eps
            elif self.parameterization == "v":
                target = self.get_v(x_0, eps, t)
            loss = F.mse_loss(predicted_eps, target.to("cuda"))
        
        pixel_tv_loss, pixel_l2_loss = torch.tensor(0, dtype=x_t.dtype, device=x_t.device), torch.tensor(0, dtype=x_t.dtype, device=x_t.device)
        if not model.probabilities < model.p_uncond and decoder is not None:
            latent_predicted_x0 = (x_t - torch.sqrt(1 - self.gather_and_expand(self.alphas_bar.to("cuda"), t.to("cuda"), x_0.shape)) * predicted_eps) / torch.sqrt(self.gather_and_expand(self.alphas_bar.to("cuda"), t.to("cuda"), x_0.shape))
            pixel_tv_loss, pixel_l2_loss = self.pixel_loss(decoder, latent_predicted_x0, input_image, t)
            del latent_predicted_x0
        del x_t, 
        gc.collect()
        torch.cuda.empty_cache()
        
        if use_attention_loss:
            return naive_loss, cwg_loss, tv_loss, dcml_loss, pixel_tv_loss, pixel_l2_loss
        return loss, pixel_tv_loss, pixel_l2_loss
    
    def _get_process_scheduling(self, reverse = True):
        if self.scheduling == 'uniform':
            diffusion_process = list(range(0, len(self.alphas_bar), self.tau)) + [len(self.alphas_bar)-1]
        elif self.scheduling == 'exp':
            diffusion_process = (np.linspace(0, np.sqrt(len(self.alphas_bar)* 0.8), self.tau)** 2)
            diffusion_process = [int(s) for s in list(diffusion_process)] + [len(self.alphas_bar)-1]
        else:
            assert 'Not Implementation'
            
        
        diffusion_process = zip(reversed(diffusion_process[:-1]), reversed(diffusion_process[1:])) if reverse else zip(diffusion_process[1:], diffusion_process[:-1])
        return diffusion_process

    @torch.no_grad()
    def invert(self, model, start_latents, cloth_agnostic_mask, densepose,
            cloth, resized_agn_mask, image_embeddings, do_cfg, time_step, batch_size):
        # time_embeddings 
        pass
        latents = start_latents.clone()
        
        diffusion_process = self._get_process_scheduling(reverse=False)
        
        for next_idx, current_idx in diffusion_process:
            # Scalar 0 -> idx - 1
            if current_idx == time_step: 
                break
            
            time = current_idx
            
            # [batch_size], [batch_size]
            current_idx = torch.Tensor([current_idx for _ in range(latents.size(0))]).long()
            next_idx = torch.Tensor([next_idx for _ in range(latents.size(0))]).long()
            
            # time_embedding: [1, 160 * 2]
            time_embedding = self.get_time_embedding(time)
            
            model_output = model(latents, cloth_agnostic_mask, densepose, cloth, resized_agn_mask, image_embeddings, time_embedding.to("cuda")
                                , is_train=False, do_cfg=do_cfg, use_attention_loss=False)  
            
            if do_cfg:
                # [batch_size, 4, Height / 8, Width / 8], [batch_size, 4, Height / 8, Width / 8]
                output_cond, output_uncond = model_output.chunk(2)
                model_output = self.cfg_scale * (output_cond - output_uncond) + output_uncond
                
            else:
                pass
            #breakpoint()
            # current_idx > next_idx의 latents를 만들었다.
            latents = (latents - torch.sqrt(1 - self.alphas_bar[current_idx].to("cuda").view(batch_size, 1, 1, 1)) * model_output) \
                * (torch.sqrt(self.alphas_bar[next_idx].to("cuda").view(batch_size, 1, 1, 1)) / torch.sqrt(self.alphas_bar[current_idx].to("cuda").view(batch_size, 1, 1, 1))) + \
                    torch.sqrt(1 - self.alphas_bar[next_idx].to("cuda").view(batch_size, 1, 1, 1)) * model_output
            
        return latents
    
    @torch.no_grad()
    def DDIM_sampling(self, model, x_T, cloth_agnostic_mask,
                      densepose, cloth, resized_agn_mask, image_embeddings, x_0=None, do_cfg=True, img_callback=None, callback_interval=50):
        # x_T: [batch_size, 4, Height / 8, Width / 8]
        # person_agnostic_mask, densepose, cloth, cloth_mask: [batch_size, 4, Height / 8, Width / 8]
        # cloth_embeddings, person_embeddings: [batch_size, 1037, 768]
        # Decoupled condition: cloth_embeddings -> SD, person_embeddings -> ControlNet
        
        diffusion_process = self._get_process_scheduling(reverse=True)
        batch_size = x_T.size()[0]
            
        x = x_T.clone()

        with tqdm(total=1000 // self.tau) as pbar:
            for prev_idx, idx in diffusion_process:
                # Scalar 999 -> 0
                time_step = idx
                
                # [batch_size], [batch_size]
                idx = torch.Tensor([idx for _ in range(x.size(0))]).long()
                prev_idx = torch.Tensor([prev_idx for _ in range(x.size(0))]).long()
                
                # time_embedding: [1, 160 * 2]
                time_embedding = self.get_time_embedding(time_step)
                
                # add DDIM inverison
                if x_0 is not None:
                    assert resized_agn_mask is not None
                    eps = torch.randn_like(x_0)
                    
                    """ Make noisy original latents W/O DDIM inversion"""
                    noisy_x_orig = self.gather_and_expand(self.sqrt_alpha_bar.to("cuda"), idx.to("cuda"), x_0.shape) * x_0 \
                        + self.gather_and_expand(self.sqrt_one_minus_alpha_bar.to("cuda"), idx.to("cuda"), x_0.shape) * eps.to("cuda")
                    
                    """ Make noisy original latents W/ DDIM inversion (More execute time...)"""
                    # noisy_x_orig = self.invert(model=model, start_latents=x_0, cloth_agnostic_mask=cloth_agnostic_mask, densepose=densepose,
                    # cloth=cloth, resized_agn_mask=resized_agn_mask, image_embeddings=image_embeddings, do_cfg=do_cfg, time_step=time_step, batch_size=batch_size)
                    
                    # masked region of resized_agn_mask set to 1
                    x = noisy_x_orig * (1. - resized_agn_mask) + resized_agn_mask * x
                
                model_output = model(x, cloth_agnostic_mask, densepose, cloth, resized_agn_mask, image_embeddings, time_embedding.to("cuda")
                                     , is_train=False, do_cfg=do_cfg, use_attention_loss=False)
                
                if do_cfg:
                    # [batch_size, 4, Height / 8, Width / 8], [batch_size, 4, Height / 8, Width / 8]
                    output_cond, output_uncond = model_output.chunk(2)
                    model_output = self.cfg_scale * (output_cond - output_uncond) + output_uncond
                    
                    if self.parameterization == "v":
                        std_cond = torch.std(output_cond, dim=[1,2,3], keepdim=True)
                        std_output = torch.std(model_output, dim=[1,2,3], keepdim=True)
                        
                        rescaled_model_output = model_output * (std_cond / std_output)
                        model_output = self.rescale * rescaled_model_output + (1 - self.rescale) * model_output
                
                else:
                    pass
                
                if self.parameterization == "eps":
                    eps = model_output 
                elif self.parameterization == "v":
                    eps = self.predict_eps_from_z_and_v(x, idx, model_output, batch_size)
                
                
                if self.parameterization == "eps":
                    # [batch_size, 4, Height / 8, Width / 8]
                    predicted_x0 = (x - torch.sqrt(1 - self.alphas_bar[idx].to("cuda").view(batch_size, 1, 1, 1)) * eps) / torch.sqrt(self.alphas_bar[idx].to("cuda").view(batch_size, 1, 1, 1))
                elif self.parameterization == "v":
                    predicted_x0 = self.predict_start_from_z_and_v(x, idx, model_output, batch_size)
                # [batch_size, 4, Height / 8, Width / 8]
                direction_pointing_to_xt = torch.sqrt(1 - self.alphas_bar[prev_idx].to("cuda")).view(batch_size, 1, 1, 1) * eps
                
                # [batch_size, 4 ,Height / 8, Width / 8]
                x = torch.sqrt(self.alphas_bar[prev_idx].to("cuda")).view(batch_size, 1, 1, 1) * predicted_x0 + direction_pointing_to_xt
                pbar.update(1)
                
                # if time_step / 100 == 1:
                #     breakpoint()
                #     torch.save(x, './latent_x.pt')
                #     torch.save(predicted_x0, './latent_predicted_x0.pt')
                #     torch.save(cloth, './latent_cloth.pt')
                #     return x #predicted_x0
                
                if time_step % callback_interval == 0:
                    if img_callback: img_callback(time_step)

        del x, direction_pointing_to_xt, model_output, eps
        gc.collect()
        torch.cuda.empty_cache()
        return predicted_x0
    
    def get_time_embedding(self, timestep):
        # Shape: (160,)
        freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
        # Shape: (1, 160)
        x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
        # Shape: (1, 160 * 2)
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


class PixelLoss(nn.Module):
    def __init__(self):
        super(PixelLoss, self).__init__()
        
        self.tv_strength = 0.0001 #0.000001 
        self.l2_strength = 0.01
        
    def forward(self, decoder, pred, gt, t):
        # [BS, C, H, W]
        pred = decoder(pred)
        
        tv_loss = self.tv_loss(pred)
        l2_loss = self.l2_loss(pred, gt)
        #logger.info(f"Time Step: {t}")
        if t == 0:
            t = torch.tensor(1)
        sqrt_t = torch.sqrt(t.to("cuda") + 1e-8)
        tv_loss, l2_loss = tv_loss * self.tv_strength, l2_loss * self.l2_strength * sqrt_t
        
        return tv_loss, l2_loss
        
    def tv_loss(self, pred):
        # pred: [BS, 3, 512, 384]
        BS, C, H, W = pred.size()
        tv_y = torch.abs(pred[:, :, 1:, :] - pred[:,:, :-1, :])
        tv_x = torch.abs(pred[:, :, :, 1:] - pred[:,:,:, :-1])
        
        loss = (tv_y.sum() + tv_x.sum()) / (BS * C * H * W)
        
        return loss
    
    def l2_loss(self, pred, gt):
        mse_loss = nn.MSELoss()
        return mse_loss(pred, gt)
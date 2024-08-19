from DINOv2 import DINOv2, MLPProjModel
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion

import model_converter
import vae_custom_converter

import torch
from peft import PeftModel

def preload_models_from_standard_weights(ckpt_path, ckpt_path2, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)
    vae_state_dict = vae_custom_converter.load_from_standard_weights_VAE(ckpt_path2, device)
    
    existing_weight = state_dict['diffusion']['unet.encoders.0.0.weight']
    weight_new_channels = torch.zeros(320,4+4+4+1,3,3)
    state_dict['diffusion']['unet.encoders.0.0.weight'] = torch.nn.Parameter(torch.cat((existing_weight, weight_new_channels), dim=1))

    diffusion = Diffusion()
    diffusion.load_state_dict(state_dict['diffusion'], strict=False)
    
    encoder = VAE_Encoder()
    encoder.load_state_dict(vae_state_dict['encoder'], strict=True)

    decoder = VAE_Decoder()
    decoder.load_state_dict(vae_state_dict['decoder'], strict=True)
    
    dinov2 = DINOv2()
    
    mlp = MLPProjModel()

    del state_dict
    
    return {
        'dinov2': dinov2,
        'mlp': mlp,
        'diffusion': diffusion,
        'encoder': encoder,
        'decoder': decoder
    }
    
def load_models_from_fine_tuned_weights(ckpt, device):
    
    state_dict = torch.load(ckpt, map_location=device, weights_only=False)["state_dict"]

    encoder = VAE_Encoder()
    decoder = VAE_Decoder()
    diffusion = Diffusion()
    dinov2 = DINOv2()
    mlp = MLPProjModel()
    
    return {
        'dinov2': dinov2,
        'mlp': mlp,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
        'state_dict': state_dict
    }
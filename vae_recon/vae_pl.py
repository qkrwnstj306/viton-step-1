import torch, os
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

# Custom

import vae_original_converter, vae_custom_converter
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from dataset import MyTrainDataset, collate_fn

def setting(original=True):

    # Load Parameter of SDv1.5
    
    if original:
        VAE_file = '../data/v1-5-pruned.ckpt'
        vae_state_dict = vae_original_converter.load_from_standard_weights(VAE_file, 'cpu')
    else:
        VAE_file = '../data/VITONHD_VAE_finetuning.ckpt'
        vae_state_dict = vae_custom_converter.load_from_standard_weights_VAE(VAE_file, 'cpu')


    # Load Dataset 
   
    data_module = MyTrainDataset(debugging=True)

    
    return vae_state_dict, data_module

def save_images(original_images, images, original, index):
    save_pth = f'./reconstruction/original_{original}'
    if not os.path.isdir(save_pth):
        os.makedirs(save_pth)
    
    # Make grid
    grid_images = torch.cat((original_images.to("cpu"), images.to("cpu")), dim=0)
    grid = make_grid(grid_images, normalize=True)
    save_path = f"{save_pth}/grid_image_index{index}_original_{original}.png"
    save_image(grid, save_path)

def main_worker(models, data_module, original=True):

    # Model Setting 

    encoder_state_dict = models["encoder"]
    decoder_state_dict = models["decoder"] 
    
    encoder = VAE_Encoder()
    encoder.load_state_dict(encoder_state_dict, strict=True)

    decoder = VAE_Decoder()
    decoder.load_state_dict(decoder_state_dict, strict=True)

    # Train and Validate 
    
    del models

    train_dataloader = torch.utils.data.DataLoader(data_module, batch_size=1, collate_fn=collate_fn, 
                                                   shuffle=False, pin_memory=True, num_workers=2)

    encoder = encoder.to("cuda")
    decoder = decoder.to("cuda")
    for index, batch in tqdm(enumerate(train_dataloader), desc="Generation", total=len(train_dataloader)):
        
        with torch.no_grad():
            input_image, cloth_agnostic_mask, densepose, cloth = batch['input_image'].to("cuda"), batch['cloth_agnostic_mask'].to("cuda"), batch['densepose'].to("cuda"), batch['cloth'].to("cuda")
            
            inputs = torch.cat((input_image, cloth_agnostic_mask, densepose, cloth), dim=0)
            encoder_noise = torch.randn((inputs.size(0), 4, 64, 48)).to("cuda")
            
            encoded_outputs = encoder(inputs, encoder_noise)
            
            decoded_ouputs = decoder(encoded_outputs)
            
        save_images(inputs, decoded_ouputs, original, index)
        


if __name__ == "__main__":
    original = False # True, False 
    models, data_module = setting(original)
    main_worker(models, data_module, original)

import torch
import torch.nn as nn
from PIL import Image
from torch.nn import functional as F

from transformers import AutoProcessor, CLIPModel
from attention import SelfAttention


class DINOv2(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        #self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    
    def forward(self, processed_image: torch.Tensor):
        # processed_image: [batch_size, 3, 518, 392]
        
        embeddings = self.model.forward_features(processed_image)
        
        # [batch_size, 3, 518, 392] -> [batch_size, 1, 1536], [batch_size, 1036, 1536 or 1024]
        CLS_embedding = embeddings['x_norm_clstoken'].unsqueeze(1)
        Patch_embedding = embeddings['x_norm_patchtokens']
        
        #image_embeddings = torch.cat((CLS_embedding, Patch_embedding), dim=1)
        
        # [batch_size, 1037, 1536]
        return Patch_embedding

class DINO_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        channels = n_head * n_embd

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)
    
    def forward(self, x):
        # x: [batch_size, 1037, 1536]
        # (Batch_Size, Height * Width + 1, Features)
        residue_short = x
        
        x = self.layernorm_1(x)
        
        x = self.attention_1(x)
        
        x += residue_short
    
        residue_short = x

        x = self.layernorm_2(x)
        
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1) 
        
        x = x * F.gelu(gate)
        
        x = self.linear_geglu_2(x)
        
        x += residue_short
        
        return x

class DINO_Transformer(nn.Module):
    def __init__(self, n_head: int, n_embd: int, layers: int):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            DINO_AttentionBlock(n_head, n_embd) for _ in range(layers)
        ])
    
    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)
        return x 

class MLPProjModel(nn.Module):
    def __init__(self, dinov2_embeddings_dim=1536):
        super().__init__()
        
        self.proj = torch.nn.ModuleList([
            torch.nn.LayerNorm(dinov2_embeddings_dim),
            DINO_Transformer(1, 1536, 4), # Change this 1536 or 1024, 4 or 5
            torch.nn.Linear(1536, 768)
        ])
        
    def forward(self, dinov2_embeddings: torch.Tensor):
        # dinov2_embeddings: [batch_size, 1037, 1536]
        
        # [batch_size, 1037, 1536] -> [batch_size, 1037, 768]
        for layer in self.proj:
            dinov2_embeddings = layer(dinov2_embeddings)
        
        # [batch_size, 1037, 768]
        return dinov2_embeddings
    
    
if __name__ == "__main__":
    
    test_dinov2 = DINOv2().to('cuda')
    test_dinov2.eval()
    
    test_proj = MLPProjModel().to('cuda')
    test_proj.train()
        
    # 6G VRAM 사용
    
    test_input = torch.zeros((1, 3, 518, 392), dtype=torch.float32).to('cuda')
    
    with torch.no_grad():
        dinov2_embeddings = test_dinov2(test_input)
        
    dinov2_embeddings = test_proj(dinov2_embeddings)
    
    print(dinov2_embeddings.size())
    

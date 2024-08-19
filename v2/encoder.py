import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    
    def __init__(self):
        super().__init__(
            # (Batch_Size, Channel, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            
            # (Batch_Size, 128, Height, Width) -> (Batch_size, 128, Height / 2, Width / 2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            # (Batch_Size, 128,  Height / 2, Width / 2) -> (Batch_Size, 256,  Height / 2, Width / 2)
            VAE_ResidualBlock(128, 256),
            
            # (Batch_Size, 256,  Height / 2, Width / 2) -> (Batch_Size, 256,  Height / 2, Width / 2)
            VAE_ResidualBlock(256, 256),
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_size, 256, Height / 4, Width / 4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            
            # (Batch_Size, 256,  Height / 4, Width / 4) -> (Batch_Size, 512,  Height / 4, Width / 4)
            VAE_ResidualBlock(256, 512),
            
            # (Batch_Size, 512,  Height / 4, Width / 4) -> (Batch_Size, 512,  Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512),
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_size, 512, Height / 8, Width / 8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            
            # (Batch_Size, 512,  Height / 8, Width / 8) -> (Batch_Size, 512,  Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),
            
            # (Batch_Size, 512,  Height / 8, Width / 8) -> (Batch_Size, 512,  Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),
            
            # (Batch_Size, 512,  Height / 8, Width / 8) -> (Batch_Size, 512,  Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),
            
            # (Batch_Size, 512,  Height / 8, Width / 8) -> (Batch_Size, 512,  Height / 8, Width / 8)
            VAE_AttentionBlock(512),
            
            # (Batch_Size, 512,  Height / 8, Width / 8) -> (Batch_Size, 512,  Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),
            
            # (Batch_Size, 512,  Height / 8, Width / 8) -> (Batch_Size, 512,  Height / 8, Width / 8)
            nn.GroupNorm(32, 512),
            
            # (Batch_Size, 512,  Height / 8, Width / 8) -> (Batch_Size, 512,  Height / 8, Width / 8)
            nn.SiLU(),
            
            # (Batch_Size, 512,  Height / 8, Width / 8) -> (Batch_Size, 8,  Height / 8, Width / 8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            
            # (Batch_Size, 8,  Height / 8, Width / 8) -> (Batch_Size, 8,  Height / 8, Width / 8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )
        
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Channel, Height, Width)
        # noise: (Batch_Size, Out_Channels, Height / 8, Width / 8)
        
        for module in self:
            # if layer has stride (2,2), add padding on x
            # getattr: object 의 attribute 를 호출한다.
            if getattr(module, 'stride', None) == (2,2):
                # We want to pad on x manually to cut it in half
                # (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom)
                # Value of padding = 1 toward Left & Top, 0 toward Right & Bottom
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
            
        # Since encoder output is mean and variance of logarithmic scale.
        # Chunk: tensor 를 쪼개는 함수이다. torch.chunk(input, 몇 개로 쪼갤지, 어떤 차원에서 적용할지)
        # (Batch_Size, 8, Height / 8, Width / 8) -> two tensors of shape (Batch_Size, 4, Height / 8 ,Width / 8)
        # Channle of mean, variance 가 4 여서, 최종 생성된 encoded feature map 은 channel = 4 이다.
        mean, log_variance = torch.chunk(x, 2, dim=1)
        
        # Clamp: min or max 값을 넘는 값이 있으면 해당 min, max 로 replace
        # Clamp the log variance between -30 and 20, so that the variance is between (circa) 1e-14 and 1e8. 
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        log_variance = torch.clamp(log_variance, -30, 20)
        
        variance = log_variance.exp()
        
        # stdev is between 3e-7 and 2e4.
        stdev = variance.sqrt()
        
        # Transform N(0, 1) -> N(mean, stdev), re-parameterization trick
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = mean + stdev * noise
        
        # Scale by a constant
        # Constant taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
        x *= 0.18215
        
        return x 
            
        
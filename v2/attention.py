import torch
from torch import nn, einsum
from torch.nn import functional as F
import math
from einops import rearrange
import time 
import os

import xformers.ops as xops

from utils import save_center_coords, save_center_coords_to_heatmap


"""
Class for Attention Loss (CWG loss & DCML loss & TV loss)
"""

class AttentionLoss(nn.Module):
    def __init__(self):
        super(AttentionLoss, self).__init__()
        self.cwg_strength = 2 # 3 
        self.dcml_strength = 0.01
        self.tv_strength = 0.0001 # 0.0001
        
        self.tv_l_type = "l2"
        
    def get_cwgloss(self, reshaped_sim, weighted_centered_grid_hw, warped_cloth_mask, mh, mw, sigma=1.0):
        """
        Calculates Center-Weighted Gaussian Loss (CWGL)

        Args:
            reshaped_sim: [BS, HW, h, w] - Attention scores
            weighted_centered_grid_hw: [BS, HW, 2] - Weighted center coordinates
            warped_cloth_mask: [BS, H, W] - Warped cloth mask (True/False)
            sigma: Standard deviation for Gaussian distribution (default 1.0)

        Returns:
            cwg_loss: Center-Weighted Gaussian Loss
        """
        BS, H, W = warped_cloth_mask.size()
        # [BS, HW, h, w]
        reshaped_sim = reshaped_sim * warped_cloth_mask.reshape(BS, H*W).contiguous().unsqueeze(2).unsqueeze(2)
        
        y_coords = torch.arange(mh, device=reshaped_sim.device)
        x_coords = torch.arange(mw, device=reshaped_sim.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Stack the coordinates along the last dimension
        # coords: [h, w, 2]
        coords = torch.stack([yy, xx], dim=-1)
        
        # [1, 1, h, w, 2] - [BS, HW, 1, 1, 2] -> [BS, HW, h, w, 2]
        differences = coords.unsqueeze(0).unsqueeze(0) - weighted_centered_grid_hw.unsqueeze(2).unsqueeze(3)
        
        # Calculate distances between attention scores and their corresponding centers
        # [BS, HW, h, w]
        distances = torch.norm(differences, dim=4)

        # Calculate Gaussian distribution probabilities, [BS, HW, h, w]
        probabilities = torch.exp(-distances / (2 * sigma**2))

        # Calculate loss, [BS, HW, h, w] -> Scalar
        cwg_loss = -torch.mean(probabilities * reshaped_sim)

        return cwg_loss * self.cwg_strength
    
    def get_tvloss(self, center_coords, mask, cH, cW):
        """
        Caculate Total Variation Loss
        Args:
            center_coords: [BS, HW, 2] - Weighted center coordinates
            mask: [BS, H, W] - Warped cloth mask (True/False)
            cH, cW: Height/Width of generated image
        Returns:
            tv_loss: Total Variation Loss
        """
        b, HW, _ = center_coords.shape
        # [b, H, W, 2]
        center_coords = center_coords.reshape(b,cH,cW,2)
        # [b, H, W, 1]
        mask = mask.unsqueeze(-1)
        y_mask = mask[:,1:] * mask[:,:-1]
        x_mask = mask[:,:,1:] * mask[:,:,:-1]

        if self.tv_l_type == "l1":
            y_tvloss = torch.abs(center_coords[:,1:] - center_coords[:,:-1]) * y_mask
            x_tvloss = torch.abs(center_coords[:,:,1:] - center_coords[:,:,:-1]) * x_mask
        
        elif self.tv_l_type == "l2":
            y_tvloss = (center_coords[:,1:] - center_coords[:,:-1])**2 * y_mask
            x_tvloss = (center_coords[:,:,1:] - center_coords[:,:,:-1])**2 * x_mask
        
        tv_loss = y_tvloss.mean() + x_tvloss.mean()
        return tv_loss * self.tv_strength
    
    def get_dcmlloss(self, center_coords, mask, cH, cW):
        """
        각 좌표 간의 거리 x 각 좌표 간의 중심좌표 차이를 maximize
        Args:
            center_coords: [BS, HW, 2] - Weighted center coordinates
            mask: [BS, H, W] - Warped cloth mask (True/False)
            cH, cW: Height/Width of generated image
        Returns:
            dcml_loss: Distance-Centering Maximization Loss
        """
        BS, H, W = mask.size()
        # [BS, H, W, 2]
        center_coords = center_coords.view(BS, H, W, 2).contiguous()
        # 좌표 생성
        coords = torch.stack(torch.meshgrid(torch.linspace(0, 1, cH, device=center_coords.device),
                                            torch.linspace(0, 1, cW, device=center_coords.device)), dim=-1).float()
        
        # 모든 좌표를 2D 텐서로 변환 [192 x 2]
        flat_coords = coords.view(-1, 2)
        # [BS, 192]
        flat_mask = mask.view(BS, -1).float()

        # 모든 좌표 쌍 간의 거리 계산 [1, 192 x 192]
        distances = torch.cdist(flat_coords, flat_coords).unsqueeze(0)
        # 하삼각형 부분 무시를 위한 마스크 생성
        triangular_mask = torch.triu(torch.ones_like(distances), diagonal=1)
        distances = distances * triangular_mask 
        
        # [BS, 16, 12 ,2] -> [BS, 192], [BS, 192]
        x_flat_values = center_coords[...,1].view(BS, -1)
        y_flat_values = center_coords[...,0].view(BS, -1)

        # 마스크를 저장할 리스트 초기화
        col_masks = []
        rows = H
        cols = W
        # 각 좌표에 대한 마스크 생성
        for i in range(rows):
            for j in range(cols):
                col_mask = torch.zeros(rows, cols, dtype=torch.bool)  # 기본값을 True로 초기화
                col_mask[i+1:, j:j+1] = True  # 해당 열까지는 False로 설정
                col_masks.append(col_mask)

        # 마스크를 (1, 192, 192) 텐서로 변환
        col_masks = torch.stack(col_masks).view(H*W,H*W).unsqueeze(0)

        # 마스크를 저장할 리스트 초기화
        row_masks = []

        # 각 좌표에 대한 마스크 생성
        for i in range(rows):
            for j in range(cols):
                row_mask = torch.zeros(rows, cols, dtype=torch.bool)  # 기본값을 True로 초기화
                row_mask[i:i+1, j+1:] = True  # 해당 행까지는 False로 설정
                row_masks.append(row_mask)

        # 마스크를 (1, 192, 192) 텐서로 변환
        row_masks = torch.stack(row_masks).view(H*W,H*W).unsqueeze(0)
        
        # 모든 원소 값 쌍 간의 차이 계산
        x_value_diffs = torch.relu(-1.0*(x_flat_values.unsqueeze(2) - x_flat_values.unsqueeze(1)))  # shape: (BS, num_coords, num_coords)
        y_value_diffs = torch.relu(-1.0*(y_flat_values.unsqueeze(2) - y_flat_values.unsqueeze(1)))  # shape: (BS, num_coords, num_coords)
        
        x_triangular_mask = torch.triu(torch.ones_like(x_value_diffs), diagonal=1)
        y_triangular_mask = torch.triu(torch.ones_like(y_value_diffs), diagonal=1)

        x_value_diffs = x_value_diffs * x_triangular_mask
        y_value_diffs = y_value_diffs * y_triangular_mask
        
        value_diffs = x_value_diffs * (row_masks).float().cuda() + y_value_diffs * (col_masks).float().cuda()
        
        # 마스크 페어 적용, [BS, 192, 192]
        mask_pairs = flat_mask.unsqueeze(2) * flat_mask.unsqueeze(1)
        
        # 거리와 원소 값 차이의 곱 계산, 같은 행 또는 열인 경우 제외
        """
        products = distances * value_diffs * mask_pairs 
        """
        products = value_diffs * mask_pairs 
        dcml_loss = -1.0 * products.mean()

        return dcml_loss * self.dcml_strength


@torch.no_grad()
def attn_mask_resize(m,h,w):
    """
    m : [BS x 1 x mask_h x mask_w] => downsample, reshape and bool, [BS x h x w]
    """  
    m = F.interpolate(m, (h, w)).squeeze(1).contiguous()
    m = torch.where(m>=0.5, True, False)
    return m

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # This combines the Wq, Wk and Wv matrices into one matrix
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        
        # This one represents the Wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        
        self.q = None
        self.k = None
        self.v = None

    def forward(self, x, causal_mask=False, attention_type = "xformers"):
        # x: # (Batch_Size, Seq_Len, Dim)

        # (Batch_Size, Seq_Len, Dim)
        input_shape = x.shape 
        
        # (Batch_Size, Seq_Len, Dim)
        batch_size, sequence_length, d_embed = input_shape 

        # (Batch_Size, Seq_Len, H, Dim / H)
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head) 

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensor of shape (Batch_Size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # -> (Batch_Size, Seq_Len, H, Dim / H)
        if attention_type=="None":
        
            # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
            q = q.view(interim_shape).transpose(1, 2)
            k = k.view(interim_shape).transpose(1, 2)
            v = v.view(interim_shape).transpose(1, 2)

            # (Batch_Size, H, Seq_Len, Dim) @ (Batch_Size, H, Dim, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
            weight = q @ k.transpose(-1, -2)
            
            if causal_mask:
                # Mask where the upper triangle (above the principal diagonal) is 1
                mask = torch.ones_like(weight, dtype=torch.bool).triu(1) 
                # Fill the upper triangle with -inf
                weight.masked_fill_(mask, -torch.inf) 
            
            # Divide by d_k (Dim / H). 
            # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
            weight /= math.sqrt(self.d_head) 

            # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
            weight = F.softmax(weight, dim=-1) 

            # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
            output = weight @ v
            
            # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
            output = output.transpose(1, 2) 

        # require: (batch_size, sequence_length, num_head, hidden_size) -> (batch_size, sequence_length, num_head, hidden_size)
        if attention_type=="xformers":
            q, k ,v = map(
            lambda t: t.unsqueeze(3)
            .reshape(batch_size, t.shape[1], self.n_heads, self.d_head)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.n_heads, t.shape[1], self.d_head)
            .contiguous(),
            (q, k, v),
            )
            
            self.q = q
            self.k = k
            self.v = v
            
            if causal_mask:
                output = xops.memory_efficient_attention(q, k, v, attn_bias=xops.LowerTriangularMask())
            else: 
                output = xops.memory_efficient_attention(q, k, v, attn_bias=None)
            
            output = (
            output.unsqueeze(0)
            .reshape(batch_size, self.n_heads, output.shape[1], self.d_head)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, output.shape[1], self.n_heads * self.d_head)
            )
            # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
            output = self.out_proj(output)

            # (Batch_Size, Seq_Len_Q, Dim_Q)
            return output
        
        if attention_type=="sdpa":
            q = q.view(interim_shape).transpose(1, 2)
            k = k.view(interim_shape).transpose(1, 2)
            v = v.view(interim_shape).transpose(1, 2)
            output = F.scaled_dot_product_attention(q, k, v, is_causal=causal_mask)
            output = output.transpose(1, 2) 
        
        # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)
        output = output.reshape(input_shape) 

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.out_proj(output) 
        
        # (Batch_Size, Seq_Len, Dim)
        return output

class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, apply_attn_loss=False, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

        self.apply_attn_loss = apply_attn_loss 
        self.idx_coords_for_debugging = -1
        
        self.q = None
        self.k = None
        self.v = None
    
    def get_attention_score(self, x, y, generated_image=True):
        input_shape = x.shape
        context_shape = y.shape
        batch_size, sequence_length, d_embed = input_shape
        _, context_sequence_length, _ = context_shape
        
        if not batch_size==1:
            x, y = x[0].unsqueeze(0), y[0].unsqueeze(0) # Get cond vector
            batch_size = 1
        
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head) 
        interim_shape_context = (batch_size, context_sequence_length, self.n_heads, self.d_head)
        
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        q = self.q_proj(x)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        k = self.k_proj(y)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        v = self.v_proj(y)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        q = q.view(interim_shape).transpose(1, 2).contiguous() 
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        k = k.view(interim_shape_context).transpose(1, 2).contiguous() 
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        v = v.view(interim_shape_context).transpose(1, 2).contiguous() 
        
        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) -> (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = q @ k.transpose(-1, -2)
        
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight /= math.sqrt(self.d_head)
        
        if generated_image:
            # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
            weight = F.softmax(weight, dim=-2)
            # (Batch_Size, H, Seq_Len_KV, Seq_Len_Q)
            weight = rearrange(weight, 'b h q kv -> b h kv q')
            width = int(math.sqrt((sequence_length * 3 / 4)))
            height = int(width * 4 / 3)
            # [batch_size, num_heads, seq_len_kv, seq_len_q] -> [batch_size, num_heads, seq_len_kv, height, width] -> [num_heads, seq_len_kv, height, width]
            attention_score = weight.reshape(batch_size, self.n_heads, context_sequence_length, height, width).contiguous().squeeze(0)
        else:
            # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
            weight = F.softmax(weight, dim=-1)
            width = 28
            height = 37
            
            attention_score = weight.reshape(batch_size, self.n_heads, sequence_length, height, width).contiguous().squeeze(0)
        
        return attention_score
    
    def forward(self, x, y, use_attention_loss = False, warped_cloth_mask = None, attention_type = "xformers"):
        # x (latent): # (Batch_Size, Seq_Len_Q, Dim_Q)
        # y (context): # (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)
        input_shape = x.shape
        context_shape = y.shape
        batch_size, sequence_length, d_embed = input_shape
        _, context_sequence_length, _ = context_shape
        
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head) 
        interim_shape_context = (batch_size, context_sequence_length, self.n_heads, self.d_head)
        
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        q = self.q_proj(x)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        k = self.k_proj(y)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        v = self.v_proj(y)
        
        # -> (Batch_Size, Seq_Len, H, Dim / H)
        if attention_type=="None":
            # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
            q = q.view(interim_shape).transpose(1, 2).contiguous() 
            # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
            k = k.view(interim_shape_context).transpose(1, 2).contiguous() 
            # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
            v = v.view(interim_shape_context).transpose(1, 2).contiguous() 
            
            # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) -> (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
            weight = q @ k.transpose(-1, -2)
            
            # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
            weight /= math.sqrt(self.d_head)
            
            # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
            weight = F.softmax(weight, dim=-1)
            
            # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV) @ (Batch_Size, H, Seq_Len_KV, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
            output = weight @ v
            
            # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H)
            output = output.transpose(1, 2).contiguous()
        
        # require: (batch_size, sequence_length, num_head, hidden_size) -> (batch_size, sequence_length, num_head, hidden_size)
        if attention_type=="xformers":
            q, k ,v = map(
            lambda t: t.unsqueeze(3)
            .reshape(batch_size, t.shape[1], self.n_heads, self.d_head)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.n_heads, t.shape[1], self.d_head)
            .contiguous(),
            (q, k, v),
            )
            
            self.q = q
            self.k = k
            self.v = v
            
            """
            attn_loss: 
            16 resolution 이 아닌 attention layer 는 use_attention_loss==True 임에도 loss 계산을 하면 안되기 때문에
            0 으로 반환한다. 
            """
            cwg_loss, tv_loss, dcml_loss = torch.tensor(0, dtype=x.dtype, device=x.device), torch.tensor(0, dtype=x.dtype, device=x.device), torch.tensor(0, dtype=x.dtype, device=x.device)
            if self.apply_attn_loss and use_attention_loss and q.size(1) == 192 and warped_cloth_mask is not None:
                sim = einsum('b i d, b j d -> b i j', q, k) * (self.d_head ** -0.5)
                # [Batch_size, HW, 1037]
                sim = sim.softmax(dim=-1)
                # [BS, HW, 1037] -> [BS, HW, 1036(37 x 28)]
                #sim = sim[:,:,1:]
                
                h = self.n_heads
                _, HW, hw = sim.shape 
                dx = int((HW//12) ** 0.5)
                mH = int(4*dx) # 16
                mW = int(3*dx) # 12
                
                assert mH == 16 and mW == 12, "You Applied Attention Loss in Incorrect Cross-Attention Layer (Set to Index 6, 7 or 8)"
                mh = 37
                mw = 28
                
                """ 
                Take attn_loss
                """
                warped_cloth_mask = attn_mask_resize(warped_cloth_mask, mH, mW) # [BS x H x W], True/False
                reshaped_sim = sim.reshape(-1, h, mH*mW, mh, mw).mean(dim=1) # [BS, h, HW, h, w] -> [BS, HW, h, w]
                
                h_linspace = torch.linspace(0,mh-1,mh, device=sim.device)
                w_linspace = torch.linspace(0,mw-1,mw, device=sim.device)
                grid_h, grid_w = torch.meshgrid(h_linspace, w_linspace)
                grid_hw = torch.stack([grid_h, grid_w]) # [2, h, w]
                
                # [BS, 768(HW), 1, 37(h), 28(w)] element-wise [1, 1, 2, 37, 28]
                weighted_grid_hw = reshaped_sim.unsqueeze(2) * grid_hw.unsqueeze(0).unsqueeze(0)  # [b HW 2 h w]
                weighted_centered_grid_hw = weighted_grid_hw.sum((-2,-1))  # [b HW 2], center-coordinate maps
                
                attn_loss_class = AttentionLoss()
                
                cwg_loss = attn_loss_class.get_cwgloss(reshaped_sim, weighted_centered_grid_hw, warped_cloth_mask, mh=mh, mw=mw, sigma=1)
                tv_loss = attn_loss_class.get_tvloss(weighted_centered_grid_hw, warped_cloth_mask, cH=mH, cW=mW)
                dcml_loss = attn_loss_class.get_dcmlloss(weighted_centered_grid_hw, warped_cloth_mask, cH=mH, cW=mW)

                """Save the center-coordinate maps for debugging"""
                self.idx_coords_for_debugging += 1
                if self.idx_coords_for_debugging % 500 == 0:
                    dir_name = "./outputs/center_coords/"
                    os.makedirs(dir_name, exist_ok=True)
                    timestamp = int(time.time())
                    file_name = dir_name + f"{self.idx_coords_for_debugging}_{timestamp}"
                    save_center_coords_to_heatmap(weighted_centered_grid_hw.view(-1, mH, mW, 2), warped_cloth_mask, file_name)
                
            output = xops.memory_efficient_attention(q, k, v, attn_bias=None)
            
            output = (
            output.unsqueeze(0)
            .reshape(batch_size, self.n_heads, output.shape[1], self.d_head)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, output.shape[1], self.n_heads * self.d_head)
            )
            # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
            output = self.out_proj(output)

            # (Batch_Size, Seq_Len_Q, Dim_Q)
            if use_attention_loss:
                return output, cwg_loss, tv_loss, dcml_loss
            return output
            
        if attention_type=="sdpa":
            q = q.view(interim_shape).transpose(1, 2).contiguous()
            k = k.view(interim_shape).transpose(1, 2).contiguous()
            v = v.view(interim_shape).transpose(1, 2).contiguous()
            output = F.scaled_dot_product_attention(q, k, v)
            output = output.transpose(1, 2) 
            
            
        # (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = output.view(input_shape)
        
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = self.out_proj(output)

        # (Batch_Size, Seq_Len_Q, Dim_Q)
        return output
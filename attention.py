import torch
from torch import nn, einsum
from torch.nn import functional as F
import math
from einops import rearrange
import time 
import os

import xformers.ops as xops
from utils import save_center_coords, save_center_coords_to_heatmap

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
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        
        self.q = None
        self.k = None
        self.v = None
    
    def get_attention_score(self, x, y, generated_image=True):
        input_shape = x.shape
        context_shape = y.shape
        batch_size, sequence_length, d_embed = input_shape
        _, context_sequence_length, _ = context_shape
        
        if not batch_size==1:
            x, y = x[0].unsqueeze(0), y[0].unsqueeze(0)
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
    
    def forward(self, x, y, attention_type = "xformers"):
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
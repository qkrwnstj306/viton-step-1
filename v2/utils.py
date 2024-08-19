import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from PIL import Image
from math import sqrt
from torchvision import transforms as T
from sklearn.manifold import TSNE
from umap import UMAP


def check_gpu_memory_usage():
    device = torch.cuda.current_device()
    allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  # 할당된 메모리를 MiB로 변환
    reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 2)  # 예약된 메모리를 MiB로 변환
    print(f"GPU {device}: Allocated memory: {allocated_memory:.2f} MiB, Reserved memory: {reserved_memory:.2f} MiB")

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params

def tensor2img(x):
    '''
    x : [BS x c x H x W] or [c x H x W]
    '''
    if x.ndim == 3:
        x = x.unsqueeze(0)
    BS, C, H, W = x.shape
    x = x.permute(0,2,3,1).reshape(-1, W, C).detach().cpu().numpy()
    # x = (x+1)/2
    # x = np.clip(x, 0, 1)
    x = np.clip(x, -1, 1)
    x = (x+1)/2
    x = np.uint8(x*255.0)
    if x.shape[-1] == 1:  # gray sclae
        x = np.concatenate([x,x,x], axis=-1)
    return x

@torch.no_grad()
def save_center_coords(weighted_centered_grid_hw, warped_cloth_mask, file_name):
    # 16 x 12 크기의 그리드 생성
    height = 16
    width = 12

    # 각 픽셀의 (a, b) 좌표값을 저장할 배열 생성
    grid = np.empty((height, width), dtype=object)
    
    """
    run dion_pl.py --resume --n_gpus 1 --batch_size 1
    weighted_centered_grid_hw.squeeze(0)[:,0]
    weighted_centered_grid_hw.squeeze(0)[:,1]
    """
    
    height_coords = weighted_centered_grid_hw.squeeze(0)[:,0].cpu().detach().numpy()
    width_coords = weighted_centered_grid_hw.squeeze(0)[:,1].cpu().detach().numpy()

    # (a, b) 좌표값을 각 픽셀에 할당
    for index in range(height * width):
        i = index // width
        j = index % width
        grid[i, j] = (height_coords[index], width_coords[index])

    # 그리드를 시각화
    plt.figure(figsize=(40, 35))
    plt.imshow(np.zeros((height, width)), cmap='gray', interpolation='nearest') # 배경 이미지

    # warped_cloth_mask 오버레이
    plt.imshow(warped_cloth_mask[0].cpu().detach().numpy() * 255., alpha=0.5) # alpha 값을 통해 투명도 조절 가능
    
    # 각 픽셀에 (a, b) 좌표값 텍스트로 표시
    for i in range(height):
        for j in range(width):
            plt.text(j, i, f'{grid[i,j]}', ha='center', va='center', color='white', fontsize=6, fontweight='bold')

    plt.title('16 x 12 Grid with (a, b) Coordinates')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.xticks(np.arange(width))
    plt.yticks(np.arange(height))
    plt.grid(False)
    plt.show()
    plt.savefig(file_name, dpi=300)
    plt.close()

@torch.no_grad()
def save_center_coords_to_heatmap(weighted_centered_grid_hw, warped_cloth_mask, file_name):
    weighted_centered_grid_hw = weighted_centered_grid_hw[0]
    warped_cloth_mask = warped_cloth_mask[0].unsqueeze(2)
    masked_center_coords = weighted_centered_grid_hw * warped_cloth_mask
    
    num_rows, num_cols = 1, 2
    
    figure = plt.figure(figsize=(20,20))
    title_lst = ["Y", "X"]
    for i, index in enumerate(title_lst):
        figure.add_subplot(num_rows, num_cols, i + 1)
        plt.imshow(masked_center_coords.squeeze()[...,i].detach().cpu().numpy(), cmap='jet')
        plt.title(f"{title_lst[i]} Grid")
        plt.colorbar()
    
    plt.show()
    plt.savefig(file_name, dpi=300)
    plt.close()
    
    
@torch.no_grad()
def visualize_and_save_features_pca(feature_maps_fit_data,feature_maps_transform_data, transform_experiments, t, save_dir, feature_type):
    feature_maps_fit_data = feature_maps_fit_data.cpu().numpy()
    pca = PCA(n_components=3)
    pca.fit(feature_maps_fit_data)
    feature_maps_pca = pca.transform(feature_maps_transform_data.cpu().numpy())  # N X 3
    feature_maps_pca = feature_maps_pca.reshape(len(transform_experiments), -1, 3)  # B x (H * W) x 3
    for i, experiment in enumerate(transform_experiments):
        pca_img = feature_maps_pca[i]  # (H * W) x 3
        w = int(sqrt(pca_img.shape[0] * 3 / 4))
        h = int(w * 4 / 3)
        if "cross_attn_k" in feature_type or "cross_attn_v" in feature_type:
            w, h = 28, 37
        
        pca_img = pca_img.reshape(h, w, 3)
        pca_img_min = pca_img.min(axis=(0, 1))
        pca_img_max = pca_img.max(axis=(0, 1))
        pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
        pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
        pca_img = T.Resize(512, interpolation=T.InterpolationMode.NEAREST)(pca_img)
        pca_img.save(os.path.join(save_dir, f"batch-{experiment}_time_{t}.png"))
        
@torch.no_grad()
def visualize_and_save_features_tsne(feature_maps_fit_data, feature_maps_transform_data, transform_experiments, t, save_dir, feature_type):
    feature_maps_fit_data = feature_maps_fit_data.cpu().numpy()
    feature_maps_transform_data = feature_maps_transform_data.cpu().numpy()

    tsne = TSNE(n_components=3)
    feature_maps_tsne = tsne.fit_transform(feature_maps_transform_data)  # N x 3

    feature_maps_tsne = feature_maps_tsne.reshape(len(transform_experiments), -1, 3)  # B x (H * W) x 3
    for i, experiment in enumerate(transform_experiments):
        tsne_img = feature_maps_tsne[i]  # (H * W) x 3
        w = int(sqrt(tsne_img.shape[0] * 3 / 4))
        h = int(w * 4 / 3)
        if "cross_attn_k" in feature_type or "cross_attn_v" in feature_type:
            w, h = 28, 37
        
        tsne_img = tsne_img.reshape(h, w, 3)
        tsne_img_min = tsne_img.min(axis=(0, 1))
        tsne_img_max = tsne_img.max(axis=(0, 1))
        tsne_img = (tsne_img - tsne_img_min) / (tsne_img_max - tsne_img_min)
        tsne_img = Image.fromarray((tsne_img * 255).astype(np.uint8))
        tsne_img = T.Resize((512), interpolation=T.InterpolationMode.NEAREST)(tsne_img)
        tsne_img.save(os.path.join(save_dir, f"batch-{experiment}_time_{t}.png"))
        
@torch.no_grad()
def visualize_and_save_features_umap(feature_maps_fit_data, feature_maps_transform_data, transform_experiments, t, save_dir, feature_type):
    feature_maps_fit_data = feature_maps_fit_data.cpu().numpy()
    feature_maps_transform_data = feature_maps_transform_data.cpu().numpy()

    umap = UMAP(n_components=3)
    feature_maps_umap = umap.fit_transform(feature_maps_transform_data)  # N x 3

    feature_maps_umap = feature_maps_umap.reshape(len(transform_experiments), -1, 3)  # B x (H * W) x 3
    for i, experiment in enumerate(transform_experiments):
        umap_img = feature_maps_umap[i]  # (H * W) x 3
        w = int(sqrt(umap_img.shape[0] * 3 / 4))
        h = int(w * 4 / 3)
        if "cross_attn_k" in feature_type or "cross_attn_v" in feature_type:
            w, h = 28, 37
        
        umap_img = umap_img.reshape(h, w, 3)
        umap_img_min = umap_img.min(axis=(0, 1))
        umap_img_max = umap_img.max(axis=(0, 1))
        umap_img = (umap_img - umap_img_min) / (umap_img_max - umap_img_min)
        umap_img = Image.fromarray((umap_img * 255).astype(np.uint8))
        umap_img = T.Resize((512), interpolation=T.InterpolationMode.NEAREST)(umap_img)
        umap_img.save(os.path.join(save_dir, f"batch-{experiment}_time_{t}.png"))
import argparse, os 
from tqdm import tqdm, tqdm
import torch
from utils import visualize_and_save_features_pca, visualize_and_save_features_tsne, visualize_and_save_features_umap
import numpy as np
from einops import rearrange

"""
python run_features_pca.py --batch_end_idx 4 --reduce_method umap --block 3 (--all_blocks True)
"""
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def load_experiments_features(feature_maps_path, block, feature_type, t, batch_idx_lst):
    feature_maps = []
    for i, batch_idx in enumerate(batch_idx_lst):
        if "self" in feature_type or "cross_attn_q" in feature_type:
            feature_map = torch.load(os.path.join(feature_maps_path, f"batch-{batch_idx}_{block}_{feature_type}_time_{t}.pt"))
            feature_map = rearrange(feature_map, 'h n d -> n (h d)')
        elif "cross" in feature_type:
            feature_map = torch.load(os.path.join(feature_maps_path, f"batch-{batch_idx}_{block}_{feature_type}_time_{t}.pt"))
            feature_map = feature_map[:,1:,:] # W/O CLS
            feature_map = rearrange(feature_map, 'h n d -> n (h d)')
        else: 
            feature_map = torch.load(os.path.join(feature_maps_path, f"batch-{batch_idx}_{block}_{feature_type}_time_{t}.pt"))[0]
            feature_map = feature_map.reshape(feature_map.shape[0], -1).t()  # N X C
        feature_maps.append(feature_map)

    return feature_maps

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--feature_pth",
        type=str,
        default="./outputs/feature_map"
    )
    
    parser.add_argument(
        "--block",
        type=str,
        default="4"
    )
    
    parser.add_argument(
        "--batch_start_idx",
        type=str,
        default="0"
    )
    
    parser.add_argument(
        "--batch_end_idx",
        type=str,
        default="1",
    )
    
    parser.add_argument(
        "--all_blocks",
        type=str2bool,
        default=False
    )
    
    parser.add_argument(
        "--reduce_method",
        type=str,
        default="pca"
    )
    
    total_steps = 1000
    time_range = np.arange(50, 1000, 50)
    iterator = tqdm(time_range, desc="visualizing features", total=total_steps)
    
    opt = parser.parse_args()
    
    feature_pth = opt.feature_pth
    block = opt.block
    start_idx = opt.batch_start_idx
    end_idx = opt.batch_end_idx
    
    print(f"Visualizing features PCA: block - {block}; feature map dir - {feature_pth}; batch range - {start_idx} ~ {end_idx}")
    
    feature_types = [
        "in_layers_features",
        "out_layers_features",
        "self_attn_q",
        "self_attn_k",
        "self_attn_v",
        "cross_attn_q",
        "cross_attn_k",
        "cross_attn_v",
    ]
    feature_pca_paths = {}
    
    if opt.reduce_method == "pca":
        visualize_and_save_features = visualize_and_save_features_pca
    elif opt.reduce_method == "tsne":
        visualize_and_save_features = visualize_and_save_features_tsne
    elif opt.reduce_method == "umap":
        visualize_and_save_features = visualize_and_save_features_umap
    pca_folder_path = f"./outputs/PCA_features_vis/{opt.reduce_method}"    
    os.makedirs(pca_folder_path, exist_ok=True)
    
    if opt.all_blocks:
        blocks = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]

        for block in blocks:
            for feature_type in feature_types:
                if "attn" in feature_type and block in ["0","1","2"]:
                        continue
                feature_pca_path = os.path.join(pca_folder_path, f"{block}_{feature_type}")
                feature_pca_paths[feature_type] = feature_pca_path
                os.makedirs(feature_pca_path, exist_ok=True)
            
            transform_experiments = np.arange(int(start_idx), int(end_idx)+1)
            
            for t in iterator:
                for feature_type in feature_types:
                    if "attn" in feature_type and block in ["0","1","2"]:
                        continue
                    fit_features = load_experiments_features(feature_pth, block, feature_type, t, transform_experiments)  # N X C
                    transform_features = load_experiments_features(feature_pth, block, feature_type, t, transform_experiments)
                    visualize_and_save_features(torch.cat(fit_features, dim=0),
                                                    torch.cat(transform_features, dim=0),
                                                    transform_experiments,
                                                    t,
                                                    feature_pca_paths[feature_type],
                                                    feature_type
                                                    )
    else:
        for feature_type in feature_types:
                if "attn" in feature_type and block in ["0","1","2"]:
                        continue
                feature_pca_path = os.path.join(pca_folder_path, f"{block}_{feature_type}")
                feature_pca_paths[feature_type] = feature_pca_path
                os.makedirs(feature_pca_path, exist_ok=True)
            
        transform_experiments = np.arange(int(start_idx), int(end_idx)+1)
        
        for t in iterator:
            for feature_type in feature_types:
                if "attn" in feature_type and block in ["0","1","2"]:
                    continue
                fit_features = load_experiments_features(feature_pth, block, feature_type, t, transform_experiments)  # N X C
                transform_features = load_experiments_features(feature_pth, block, feature_type, t, transform_experiments)
                visualize_and_save_features(torch.cat(fit_features, dim=0),
                                                torch.cat(transform_features, dim=0),
                                                transform_experiments,
                                                t,
                                                feature_pca_paths[feature_type],
                                                feature_type
                                                )


if __name__ == "__main__":
    main()
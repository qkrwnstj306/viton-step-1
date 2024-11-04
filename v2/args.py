import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def arguments():
    parser = argparse.ArgumentParser(description='Train Arguments')

    # train
    parser.add_argument('-lr', '--lr', type=float, default=1e-5)
    parser.add_argument('-e', '--n_epochs', type=int, default=1000)
    parser.add_argument('-fp16', '--do_fp16', type=str2bool, default=True)
    parser.add_argument('-a', '--accumulation_steps', type=int, default=2) # 일반적으로는 bs: 32 (8 x 4 GPUs)로 학습하니 64 (32 x 2 a)의 batch_size로 생각 
    parser.add_argument('-w', '--batch_frequency', type=int, default=1001)
    parser.add_argument('-sc', '--scheduler', type=str2bool, default=False)
    parser.add_argument('--tb_save_dir', type=str, default='./log')
    parser.add_argument('--sd_locked', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--use_attention_loss', type=str2bool, default=False)
    parser.add_argument('--use_pixel_loss', type=str2bool, default=False)
    parser.add_argument('--weight_dir', type=str, default='eps')
    
    # all inference 
    parser.add_argument('--paired', type=str2bool, default=True)
    parser.add_argument('--blending', type=str2bool, default=False)
    parser.add_argument('--save_dir', type=str, default='./outputs/epoch-0')
    
    # attribution map visualization
    parser.add_argument('--generated_image', type=str2bool, default=True)
    parser.add_argument('--per_head', type=str2bool, default=False) # per_time == False 여야 사용 가능
    parser.add_argument('--per_time', type=str2bool, default=False) # per_head == False 여야 사용 가능 
    # generated_image, per_time, per_head == False 로 setting.
    parser.add_argument('--specific_reference_attribution_maps', type=str2bool, default=False) 
    parser.add_argument('--cloth_pixel_visualization', type=str2bool, default=True)
    parser.add_argument('--patch_index', type=int, default=0) # [0,15]
    parser.add_argument('--cal_IMACS', type=str2bool, default=False) 
    # IMACS for reference attribution map이 잘 안나올때, cloth mask 에 해당하는 영역만 score 보기
    parser.add_argument('--cloth_forcing', type=str2bool, default=False) 
    # 오직 하나의 데이터에 대해서만 attention inference
    parser.add_argument('--only_one_data', type=str2bool, default=False) 
    parser.add_argument('--certain_data_idx', type=str, default="00865_00.jpg") 
    
    # train & inference
    parser.add_argument('-bs', '--batch_size', type=int, default=1)
    parser.add_argument('-cfg', '--do_cfg', type=str2bool, default=True)
    parser.add_argument('-ng', '--n_gpus', type=int, default=4)
    parser.add_argument('--cfg_scale', type=float, default=5) # if set 1, only use cond 
    parser.add_argument('--parameterization', type=str, default='eps') # v
    parser.add_argument('--rescale', type=float, default=0.0) # 0이면 그냥 CFG,
    parser.add_argument('-s', '--seed', type=int, default=23) 
    
    # debugging
    parser.add_argument('--debugging', type=str2bool, default=False)
    
    # for extract_feature_map.py, not run_features_pca.py
    parser.add_argument('--block_index', type=int, default=4) # [0,11], if set -1, all index
    
    args = parser.parse_args()
    return args
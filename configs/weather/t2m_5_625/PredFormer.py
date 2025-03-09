method = 'PredFormer'

model_config = {
    # image h w c
    'height': 32,
    'width': 64,
    'num_channels': 1,
    # video length in and out
    'pre_seq': 12,
    'after_seq': 12,
    # patch size
    'patch_size': 4,
    'dim': 256, 
    'heads': 8,
    'dim_head': 32,
    # dropout
    'dropout': 0.1,
    'attn_dropout': 0.1,
    'drop_path': 0.25,
    'scale_dim': 2,
    # depth
    'depth': 1,
    'Ndepth': 2, # For FullAttention-8, for BinaryST, BinaryST, FacST, FacTS-4, for TST,STS-3, for TSST, STTS-2
}
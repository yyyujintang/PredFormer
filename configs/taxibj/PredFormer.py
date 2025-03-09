method = 'PredFormer'

model_config = {
    # image h w c
    'height': 32,
    'width': 32,
    'num_channels': 2,
    # video length in and out
    'pre_seq': 4,
    'after_seq': 4,
    # patch size
    'patch_size': 4,
    'dim': 256, 
    'heads': 8,
    'dim_head': 32,
    # dropout
    'dropout': 0.1,
    'attn_dropout': 0.1,
    'drop_path': 0.1,
    'scale_dim': 4,
    # depth
    'depth': 1,
    'Ndepth': 2, # For FullAttention-8, for BinaryST, BinaryST, FacST, FacTS-4, for TST,STS-3, for TSST, STTS-2

}

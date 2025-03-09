method = 'PredFormer'

model_config = {
    # image h w c
    'height': 256,
    'width': 256,
    'num_channels': 3,
    # video length in and out
    'pre_seq': 4,
    'after_seq': 4,
    # patch size
    'patch_size': 8,
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
    'Ndepth': 3, # For FullAttention-12, for BinaryST, BinaryST, FacST, FacTS-6, for TST,STS-4, for TSST, STTS-3

}

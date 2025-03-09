export CUDA_VISIBLE_DEVICES=0
CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
EX_NAME="mmnist/${CURRENT_TIME}_PredFormer_depth8_TSST_sd0.0_dp0.0_ps8_bs16_256_8_32_1e-3_Adamw_onecycle_2000ep"

python tools/train.py \
    --config_file configs/mmnist/PredFormer.py \
    --dataname mmnist \
    --data_root data \
    --res_dir work_dirs \
    --batch_size 16 \
    --epoch 2000 \
    --overwrite \
    --lr 1e-3 \
    --opt adamw \
    --weight_decay 1e-2 \
    --ex_name "$EX_NAME"  \
    --tb_dir logs_tb/03_08
export CUDA_VISIBLE_DEVICES=0
CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
EX_NAME="taxibj/${CURRENT_TIME}_PredFormer_depth2_STS_sd0.1_dp0.1_ps8_bs16_256_8_32_1e-3_Adamw_onecycle_200ep"

python tools/train.py \
    --config_file configs/taxibj/PredFormer.py \
    --dataname taxibj \
    --data_root data \
    --res_dir work_dirs \
    --batch_size 16 \
    --epoch 200 \
    --overwrite \
    --lr 1e-3 \
    --opt adamw \
    --weight_decay 1e-2 \
    --ex_name "$EX_NAME"  \
    --tb_dir logs_tb/03_08
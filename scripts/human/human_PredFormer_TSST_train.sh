export CUDA_VISIBLE_DEVICES=0
CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
EX_NAME="human/${CURRENT_TIME}_PredFormer_depth3_TSST_sd0.1_dp0.1_256_8_32_lr1e-3_50ep_cos_bs8_ps8_Adamw"

python tools/train.py \
    --config_file configs/human/PredFormer.py \
    --dataname human \
    --data_root data \
    --res_dir work_dirs \
    --batch_size 8 \
    --epoch 50 \
    --sched cosine \
    --warmup_epoch 0 \
    --overwrite \
    --lr 1e-3 \
    --opt adamw \
    --weight_decay 1e-2 \
    --ex_name "$EX_NAME"  \
    --tb_dir logs_tb/03_08
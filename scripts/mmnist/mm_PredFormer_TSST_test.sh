export CUDA_VISIBLE_DEVICES=0
CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
EX_NAME="mmnist/mm_PredFormer_TSST_MSE12.4"

python tools/test.py \
    --config_file configs/mmnist/PredFormer.py \
    --dataname mmnist \
    --data_root data \
    --res_dir work_dirs/ \
    --batch_size 16 \
    --ex_name "$EX_NAME"  \

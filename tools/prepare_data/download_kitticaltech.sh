#!/usr/bin/env bash

# run this script on the root
mkdir -p data/kitti_hkl
cd data

# you can download kitti and caltech datasets from Baidu Cloud `https://pan.baidu.com/s/1fudsBHyrf3nbt-7d42YWWg?pwd=kjfk`
#  or run the following scripts

wget https://www.dropbox.com/s/rpwlnn6j39jjme4/kitti_data.zip?dl=0 -O kitti_data.zip
unzip kitti_data.zip -d kitti_hkl/

# https://drive.google.com/file/d/1Ff7u5guRxsDMcNx402eEQVRBF5Nor-Ua/view?usp=sharing
# https://drive.google.com/file/d/1c0kUYGNQ_zmkTdLmYd04yPVePHVDlKAF/view?usp=sharing
# https://drive.google.com/file/d/1c0kUYGNQ_zmkTdLmYd04yPVePHVDlKAF/view?usp=sharing
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1Ff7u5guRxsDMcNx402eEQVRBF5Nor-Ua' -O indices_cache.npy
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1c0kUYGNQ_zmkTdLmYd04yPVePHVDlKAF' -O data_cache.npy
# 只用了caltech的test部分做测试
wget https://data.caltech.edu/records/f6rph-90m20/files/data_and_labels.zip?download=1 -O caltech_full.zip
mkdir caltech_full
unzip caltech_full.zip -d caltech_full/
mv caltech_full/Test caltech
rm -r caltech_full

echo "finished"


# Download and arrange them in the following structure:
# OpenSTL
# └── data
#     ├── caltech
#     │   ├── set06
#     │   ├── ...
#     │   ├── data_cache.npy  # from Baidu Cloud (the cached version of caltech)
#     │   ├── indices_cache.npy  # from Baidu Cloud 
#     ├── kitti_hkl
#     │   ├── sources_train.hkl
#     │   ├── ...
#     │   ├── X_val.hkl

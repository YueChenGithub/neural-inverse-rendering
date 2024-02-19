import os

configs = ['./config/dtu/lego_3072.ini']

for config in configs:
    os.system(f"python train.py --config {config} --max_epoch 30")
    os.system(f"python cal_correction.py --config {config}")
    os.system(f"python test.py --config {config} --relighting False --test_val_dataset False")
    os.system(f"python test.py --config {config} --relighting True --test_val_dataset False")

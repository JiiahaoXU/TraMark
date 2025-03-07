device_num=0
seed=0

CUDA_VISIBLE_DEVICES=$device_num python federated.py --data cifar10 --aggr tramark --seed $seed --alpha 0.02
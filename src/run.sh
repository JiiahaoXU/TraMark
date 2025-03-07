device_num=0
seed=0

CUDA_VISIBLE_DEVICES=$device_num python federated.py --aggr tramark --data cifar10 --alpha 0.02 --k 0.01 --non_iid --gamma 0.5
device_num=1
seed=0


# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data fmnist --aggr tramark_w_warmup --seed $seed --warmup_rounds 0.7 --non_iid --alpha 0.5

# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data fmnist --aggr avg --seed $seed --non_iid --alpha 0.1
# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data fmnist --aggr avg --seed $seed --non_iid --alpha 0.2
# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data fmnist --aggr avg --seed $seed --non_iid --alpha 0.3
# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data fmnist --aggr avg --seed $seed --non_iid --alpha 0.4

# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data cifar10 --aggr avg --seed $seed --non_iid --alpha 0.1
# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data cifar10 --aggr avg --seed $seed --non_iid --alpha 0.2
# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data cifar10 --aggr avg --seed $seed --non_iid --alpha 0.3
# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data cifar10 --aggr avg --seed $seed --non_iid --alpha 0.4

CUDA_VISIBLE_DEVICES=$device_num python federated.py --data cifar100 --aggr avg --seed $seed --non_iid --alpha 0.1
# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data cifar100 --aggr avg --seed $seed --non_iid --alpha 0.2
# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data cifar100 --aggr avg --seed $seed --non_iid --alpha 0.3
# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data cifar100 --aggr avg --seed $seed --non_iid --alpha 0.4

# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data tiny --aggr avg --seed $seed --non_iid --alpha 0.1
# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data tiny --aggr avg --seed $seed --non_iid --alpha 0.2
# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data tiny --aggr avg --seed $seed --non_iid --alpha 0.3
# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data tiny --aggr avg --seed $seed --non_iid --alpha 0.4

# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data cifar10 --aggr avg --seed $seed --non_iid --alpha 0.1
# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data cifar100 --aggr avg --seed $seed --non_iid --alpha 0.1
# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data tiny --aggr avg --seed $seed --non_iid --alpha 0.1


# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data fmnist --aggr tramark_w_warmup --seed $seed --warmup_rounds 0.5 --non_iid --alpha 0.3
# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data cifar10 --aggr tramark_w_warmup --seed $seed --warmup_rounds 0.5 --non_iid --alpha 0.3
# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data cifar100 --aggr tramark_w_warmup --seed $seed --warmup_rounds 0.5 --non_iid --alpha 0.3
# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data tiny --aggr tramark_w_warmup --seed $seed --warmup_rounds 0.5 --non_iid --alpha 0.3

# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data fmnist --aggr tramark_w_warmup --seed $seed --warmup_rounds 0.5 --non_iid --alpha 0.4
# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data cifar10 --aggr tramark_w_warmup --seed $seed --warmup_rounds 0.5 --non_iid --alpha 0.4
# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data cifar100 --aggr tramark_w_warmup --seed $seed --warmup_rounds 0.5 --non_iid --alpha 0.4
# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data tiny --aggr tramark_w_warmup --seed $seed --warmup_rounds 0.5 --non_iid --alpha 0.4




# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data tiny --aggr avg --seed $seed --num_workers 2 --non_iid --alpha 0.5
# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data cifar100 --aggr avg --seed 1

# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data cifar100 --aggr avg --seed 2

# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data cifar100 --aggr fedtracker --seed $seed --non_iid --alpha 0.5




# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data $dataset --aggr avg \
# --num_agents 10 --agent_frac 1.0 --snap 1 \
# --bs 64 --local_ep 5 --round 100 --client_lr 0.01 --momentum 0.9 --seed $seed

# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data $dataset --aggr tramark \
# --num_agents 10 --agent_frac 1.0 --snap 1 \
# --bs 64 --local_ep 5 --round 100 --client_lr 0.01 --momentum 0.9 --seed $seed

# CUDA_VISIBLE_DEVICES=$device_num python federated.py --data $dataset --aggr tramark_w_warmup \
# --num_agents 10 --agent_frac 1.0 --snap 1 \
# --bs 64 --local_ep 5 --round 100 --client_lr 0.01 --momentum 0.9 --warmup_rounds 0.2 --seed $seed






###### FMNIST -> 50 round is totally enough.


# CUDA_VISIBLE_DEVICES=$device_num python pruning_attack.py --data cifar10 --aggr our \
# --num_agents 10 --agent_frac 1.0 --snap 1 \
# --model_path /home/jiahaox/Watermark-FL/src/logs/cifar10/aggr_our/2025-02-01-01-30/best_models.pt

# CUDA_VISIBLE_DEVICES=$device_num python finetuning_attack.py --data cifar10 --aggr our \
# --num_agents 10 --agent_frac 1.0 --snap 1 --client_lr 0.001 --local_ep 5 \
# --model_path /home/jiahaox/Watermark-FL/src/logs/cifar10/aggr_our/2025-02-01-01-30/best_models.pt

# CUDA_VISIBLE_DEVICES=$device_num python pruning_attack.py --data tiny --aggr our \
# --num_agents 10 --agent_frac 1.0 --snap 1 --client_lr 0.001 --local_ep 5 \
# --model_path /home/jiahaox/Watermark-FL/src/logs/tiny/aggr_our/2025-02-02-00-14/best_models.pt


# CUDA_VISIBLE_DEVICES=7 python federated.py --data fmnist --aggr our \
# --num_agents 100 --agent_frac 1.0 \
# --bs 64 --local_ep 5 --round 100







# FedCRMW: Federated model ownership verification with compression-resistant model watermarking use only 10 clients
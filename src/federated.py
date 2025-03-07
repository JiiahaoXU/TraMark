import math
import copy
import models
import numpy as np
from agent import Agent
from aggregation import Aggregation
import torch
import random
from torch.utils.data import DataLoader
import torch.nn as nn
import logging
import argparse
from utils import setup_logging, get_datasets, distribute_data_dirichlet, distribute_data

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='pass in a parameter')
    
    parser.add_argument('--data', type=str, default='cifar100',
                        help="dataset we want to train on")
    
    parser.add_argument('--num_agents', type=int, default=10,
                        help="number of agents:K")
    
    parser.add_argument('--agent_frac', type=float, default=1.0,
                        help="fraction of agents per round:C")
    
    parser.add_argument('--local_ep', type=int, default=5,
                        help="number of local epochs:E")
    
    parser.add_argument('--bs', type=int, default=64,
                        help="local batch size: B")
    
    parser.add_argument('--client_lr', type=float, default=0.01,
                        help='clients learning rate')
    
    parser.add_argument('--snap', type=int, default=1,
                        help="do inference in every num of snap rounds")

    parser.add_argument('--device',  default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help="To use cuda, set to a specific GPU ID.")

    parser.add_argument('--num_workers', type=int, default=2, 
                        help="num of workers for multithreading")

    parser.add_argument('--non_iid', action='store_true', default=False)

    parser.add_argument('--debug', action='store_true', default=False)

    parser.add_argument('--gamma',type=float, default=0.5)

    parser.add_argument('--aggr', type=str, default='waffle', choices=['avg', 'tramark'],
                        help="aggregation function to aggregate agents' local weights")
    
    parser.add_argument('--lr_decay',type=float, default=0.99)

    parser.add_argument('--momentum',type=float, default=0.9)

    parser.add_argument('--wd', type=float, default=1e-4)

    parser.add_argument('--alpha', type=float, default=0.1)

    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--k', type=float, default=0.01)



    args = parser.parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    round_dict = {'fmnist': 50, 
                  'cifar10': 100,
                  'cifar100': 100,
                  'tiny': 50
                  }
    
    args.rounds = round_dict[args.data]

    args.log_dir = setup_logging(args)

    train_dataset, val_dataset = get_datasets(args.data)

    if args.data == "cifar100":
        num_target = 100
    elif args.data in ['cifar10', 'fmnist']:
        num_target = 10
    elif args.data == 'tiny':
        num_target = 200
        

    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    if args.non_iid:
        user_groups = distribute_data_dirichlet(train_dataset, args)
    else:
        user_groups = distribute_data(train_dataset, args, n_classes=num_target)

    global_model = models.get_model(args.data).to(args.device)


    agents, agent_data_sizes = [], {}
    for _id in range(0, args.num_agents):
        agent = Agent(_id, args, train_dataset, user_groups[_id])
        agent_data_sizes[_id] = agent.n_data
        agents.append(agent)

    aggregator = Aggregation(agent_data_sizes, args, global_model, val_loader)

    global_model_list = {}
    for i in range(args.num_agents):
        global_model_list[i] = copy.deepcopy(global_model)

    criterion = nn.CrossEntropyLoss().to(args.device)

    best_acc = -1

    for rnd in range(1, args.rounds + 1):
        logging.info("--------round {} ------------".format(rnd))

        chosen = [i for i in range(args.num_agents)]

        for agent_id in chosen:
            global_model = global_model_list[agent_id]
            new_local_model = agents[agent_id].local_train(global_model, criterion, rnd)
            global_model_list[agent_id] = copy.deepcopy(new_local_model)

        global_model_list = aggregator.aggregate_updates(global_model, global_model_list, rnd)


    logging.info('Best results:')
    logging.info('ACC:              %.2f' % aggregator.best_acc)
    logging.info('Veri:             %.2f' % aggregator.best_veri)
    logging.info('Training has finished!')

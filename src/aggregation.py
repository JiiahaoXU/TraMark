from torch.nn.utils import parameters_to_vector
import logging
from utils import vector_to_model
from watermark_utils import get_watermark_dataset, generate_masks_topk, masks_to_vector, test_model
import torch.nn as nn
import torch.optim as optim
import torch
import copy
import os
import builtins


class Aggregation():
    def __init__(self, agent_data_sizes, args, global_model, main_task_test_loader):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.main_task_test_loader = main_task_test_loader
        self.best_acc = -1
        self.best_veri = -1


    def aggregate_updates(self, global_model, agent_updates_dict=None, round=None):
        
        if self.args.aggr == 'avg':    
            aggregated_models_dict = self.agg_avg(agent_updates_dict, round)
            return aggregated_models_dict

        if self.args.aggr == 'tramark':
            aggregated_models_dict = self.agg_tramark(agent_updates_dict, round)
            return aggregated_models_dict



    def agg_avg(self, agent_models_dict, round):
        """ classic fed avg: Weighted average based on data size."""

        sm_updates, total_data = 0, 0

        for agent_id, local_model in agent_models_dict.items():
            local_vector = parameters_to_vector(
                    [local_model.state_dict()[name] for name in local_model.state_dict()]
                ).detach()
            n_agent_data = self.agent_data_sizes[agent_id]
            sm_updates += local_vector * n_agent_data  
            total_data += n_agent_data

        sm_updates /= total_data

        aggregated_models_dict = {}
        for agent_id, local_model in agent_models_dict.items():
            new_model = copy.deepcopy(local_model)
            new_model = vector_to_model(sm_updates, new_model)  
            aggregated_models_dict[agent_id] = new_model
        
        if round % self.args.snap == 0:
            
            # logging.info("Training round %d" % round)
            accuracy = test_model(new_model, self.main_task_test_loader)
            logging.info("Main task accuracy: %.2f%%" % accuracy)

            if self.best_acc < accuracy:
                self.best_acc = accuracy
        return aggregated_models_dict


    def initialize_mask(self, global_model):
        self.class_loaders = get_watermark_dataset(self.args)
        self.main_task_mask, self.watermarking_mask = generate_masks_topk(partition_ratio=self.args.k, model=global_model)
        self.criterion = nn.CrossEntropyLoss()
        self.main_task_mask_flat = masks_to_vector(self.main_task_mask).cuda()
        self.watermarking_mask_flat = masks_to_vector(self.watermarking_mask).cuda()


    def agg_tramark(self, agent_models_dict, round):

        """
        1. Get the backdoor dataset
        2. Recover local model
        3. Perform individual backdoor injection to masked local model, save it
        4. Aggregate the masked part
        5. Return each personalized local model to local client.
        """

        if round < int(self.args.rounds * self.args.alpha):
            self.current_global_model = copy.deepcopy(list(agent_models_dict.values())[0])
            return self.agg_avg(agent_models_dict, round)
        
        if round == int(self.args.rounds * self.args.alpha):
            self.initialize_mask(self.current_global_model)
            

        aggregated_vector = torch.zeros_like(self.main_task_mask_flat, dtype=torch.float32).cuda()
        total_data = 0
        for agent_id, local_model in agent_models_dict.items():
            local_vector = parameters_to_vector(
                    [local_model.state_dict()[name] for name in local_model.state_dict()]
                ).detach()
            n_agent_data = self.agent_data_sizes[agent_id]
            aggregated_vector += local_vector * n_agent_data  
            total_data += n_agent_data

        aggregated_vector /= total_data

        aggregated_models_dict = {}

        for agent_id, local_model in agent_models_dict.items():

            local_vector = parameters_to_vector(
                        [local_model.state_dict()[name] for name in local_model.state_dict()]
                    ).detach()
            
            ########### Masked aggregation ###########
            final_vector = self.main_task_mask_flat * aggregated_vector + self.watermarking_mask_flat * local_vector

            new_model = copy.deepcopy(local_model)
            new_model = vector_to_model(final_vector, new_model)
            aggregated_models_dict[agent_id] = new_model

        main_task_acc = []

        print('Watermarking process starts')

        ########### Watermark Injection ###########

        for id, local_model in aggregated_models_dict.items():
            if round % self.args.snap == 0:
                pre_inject_acc = test_model(local_model, self.main_task_test_loader)
    
            optimizer = optim.SGD(local_model.parameters(), lr=0.0001, momentum=0.0)

            local_model.train()
            for epoch in range(5):
                
                for inputs, targets in self.class_loaders[id]['train']:
                    
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                    outputs = local_model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()

                    ########### Zeroing out gradients in the main task region ###########
                    for name, param in dict(local_model.named_parameters()).items():
                        if name in self.watermarking_mask:
                            if param.grad is not None:
                                param.grad.data.mul_(self.watermarking_mask[name].cuda())

                    optimizer.step()

                print(f"Client {id}, watermarking epoch {epoch + 1}/5 finished.")
            if round % self.args.snap == 0:
                post_injection_acc = test_model(local_model, self.main_task_test_loader)
                logging.info(f"Client %d, accuracy changes before and after injection: %.2f%% --> %.2f%%" % (id, pre_inject_acc, post_injection_acc))
                main_task_acc.append(post_injection_acc)

            aggregated_models_dict[id] = copy.deepcopy(local_model)

        logging.info('-----'*4)

        verification = []
        if round % self.args.snap == 0:
            
            current_acc = sum(main_task_acc) / len(main_task_acc)

            if self.args.data in ['tiny', 'fmnist'] or round >= self.args.rounds // 2:
                logging.info('--------Model leaker verification--------')

                for id, local_model in aggregated_models_dict.items():
                    local_model.eval()

                    current_acc_trigger = []
                    for i in range(10):
                        acc = test_model(local_model, self.class_loaders[i]['test'])
                        current_acc_trigger.append(builtins.round(acc, 2))

                    max_index = current_acc_trigger.index(max(current_acc_trigger))

                    if max_index == id:
                        verification.append(1)
                        logging.info('Client id: %d, accuracy on trigger set are %s, verficication success!' % (id, str(current_acc_trigger)))
                    else:
                        verification.append(0)
                        logging.info('Client id: %d, accuracy on trigger set are %s, verficication fail!' % (id, str(current_acc_trigger)))

                current_veri = sum(verification) / len(verification) * 100
                self.best_veri = current_veri


                logging.info('Avg VR: %.2f%%' % (current_veri))

                if self.best_acc < current_acc:
                    self.best_acc = current_acc
                    
            logging.info('Avg ACC: %.2f%%' % (current_acc))
                
        return aggregated_models_dict


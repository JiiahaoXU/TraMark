from torch.nn.utils import parameters_to_vector
import logging
from utils import vector_to_model, save_bn_params, rewind_bn_params
from get_backdoor import get_backdoor, generate_masks, freeze_bn, masks_to_vector, test_model, unfreeze_bn, generate_masks_topk
import torch.nn as nn
import torch.optim as optim
import torch
import copy
import os
import builtins
from fedtracker import *


class Aggregation():
    def __init__(self, agent_data_sizes, args, n_params, global_model, main_task_test_loader):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.server_lr = args.server_lr
        self.main_task_test_loader = main_task_test_loader

        self.best_acc = -1
        self.best_veri = -1
        
        if self.args.aggr == 'tramark':
            self.class_loaders = get_backdoor(self.args)
            self.super_mask, self.additional_mask = generate_masks(ratio=0.99, model=global_model)
            # self.super_mask, self.additional_mask = self.super_mask.cuda(), self.additional_mask.cuda()
            self.criterion = nn.CrossEntropyLoss()
            self.super_mask_flat = masks_to_vector(self.super_mask).cuda()
            self.additional_mask_flat = masks_to_vector(self.additional_mask).cuda()
            # xx

        if self.args.aggr == 'waffle':
            self.learning_rate = 0.00001
            # # 0.0001 will destroy the convergence.
            if self.args.data == 'cifar100':
                self.learning_rate = self.learning_rate / 10


            self.class_loaders = get_backdoor(self.args)
            self.criterion = nn.CrossEntropyLoss()

        if self.args.aggr == 'fedtracker' or self.args.aggr == 'test':
            # snew_model = copy.deepcopy(local_model)
            # new_model = vector_to_model(sm_updates, new_model)
            self.name_dict = {'fmnist': 'fc1', 'cifar10': "classifier.6", 'cifar100': 'classifier.3', 'tiny': 'head'}
            self.weight_size = get_embed_layers_length(copy.deepcopy(global_model), self.name_dict[self.args.data])
            # print(weight_size)
            if self.args.data == 'cifar100':
                bit_length = 4
            else:
                bit_length = 128
            self.local_fingerprints = generate_fingerprints(self.args.num_agents, bit_length)

            self.extracting_matrices = generate_extracting_matrices(self.weight_size, bit_length, self.args.num_agents)

    def aggregate_updates(self, global_model, agent_updates_dict=None, round=None):
        
        cur_global_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]
        ).detach()

        if self.args.aggr == 'avg':    
            aggregated_models_dict = self.agg_avg(agent_updates_dict, round)
            return aggregated_models_dict

        if self.args.aggr in ['tramark', 'tramark_w_warmup']:
            aggregated_models_dict = self.agg_tramark(agent_updates_dict, cur_global_params, round)
            return aggregated_models_dict

        if self.args.aggr == 'waffle':
            aggregated_models_dict = self.agg_waffle(agent_updates_dict, round)

            # aggregated_models_dict = self.agg_avg(agent_updates_dict, round)

            return aggregated_models_dict

        if self.args.aggr == 'fedtracker':    
            aggregated_models_dict = self.agg_fedtracker(agent_updates_dict, round)
            return aggregated_models_dict
        

        if self.args.aggr == 'test':    
            aggregated_models_dict = self.agg_test(agent_updates_dict, round)
            return aggregated_models_dict

        # new_global_params = (cur_global_params + self.server_lr * aggregated_updates).float()
        # global_model = vector_to_model(new_global_params, global_model)
        
        # return global_model


    def agg_avg(self, agent_models_dict, round):
        """ classic fed avg: Weighted average based on data size."""

        sm_updates, total_data = 0, 0
        # for _id, update in agent_updates_dict.items():
        #     n_agent_data = self.agent_data_sizes[_id]
        #     sm_updates +=  n_agent_data * update
        #     total_data += n_agent_data

        for agent_id, local_model in agent_models_dict.items():
            local_vector = parameters_to_vector(
                    [local_model.state_dict()[name] for name in local_model.state_dict()]
                ).detach()
            n_agent_data = self.agent_data_sizes[agent_id]
            sm_updates += local_vector * n_agent_data  # 累加所有 agent 的参数
            total_data += n_agent_data

        # 对 super_mask 部分的参数取均值
        # aggregated_vector = aggregated_vector / len(agent_models_dict)
        sm_updates /= total_data

        aggregated_models_dict = {}
        for agent_id, local_model in agent_models_dict.items():
            new_model = copy.deepcopy(local_model)
            new_model = vector_to_model(sm_updates, new_model)  # 假设你有一个 vector_to_model 函数
            aggregated_models_dict[agent_id] = new_model
        
        if round % self.args.snap == 0:
            
            # logging.info("Training round %d" % round)
            accuracy = test_model(new_model, self.main_task_test_loader)
            logging.info("Main task accuracy: %.2f%%" % accuracy)

            if self.best_acc < accuracy:
                self.best_acc = accuracy
        return aggregated_models_dict
    
    
    def agg_fedtracker(self, agent_models_dict, round):
        """ classic fed avg: Weighted average based on data size."""

        sm_updates, total_data = 0, 0
        # for _id, update in agent_updates_dict.items():
        #     n_agent_data = self.agent_data_sizes[_id]
        #     sm_updates +=  n_agent_data * update
        #     total_data += n_agent_data

        for agent_id, local_model in agent_models_dict.items():
            local_vector = parameters_to_vector(
                    [local_model.state_dict()[name] for name in local_model.state_dict()]
                ).detach()
            n_agent_data = self.agent_data_sizes[agent_id]
            sm_updates += local_vector * n_agent_data  # 累加所有 agent 的参数
            total_data += n_agent_data

        # 对 super_mask 部分的参数取均值
        # aggregated_vector = aggregated_vector / len(agent_models_dict)
        sm_updates /= total_data

        aggregated_models_dict = {}
        tvc = []
        acc = []
        for client_idx, local_model in agent_models_dict.items():
            new_model = copy.deepcopy(local_model)
            new_model = vector_to_model(sm_updates, new_model)  # 假设你有一个 vector_to_model 函数

            #local fingerprint insertion
            client_fingerprint = self.local_fingerprints[client_idx]
            embed_layers = get_embed_layers(new_model, self.name_dict[self.args.data])
            fss, extract_idx = extracting_fingerprints(embed_layers, self.local_fingerprints, self.extracting_matrices)

            count = 0
            while (extract_idx != client_idx or (client_idx == extract_idx and fss < 0.85))  and count <= 5:
                client_grad = calculate_local_grad(embed_layers,
                                                client_fingerprint,
                                                self.extracting_matrices[client_idx])
                client_grad = torch.mul(client_grad, -0.0005)
                weight_count = 0
                for embed_layer in embed_layers:
                    weight_length = embed_layer.weight.shape[0]
                    # print(weight_length)
                    # print(client_grad)

                    embed_layer.weight.data += client_grad[weight_count: weight_count + weight_length].data.cuda()
                    weight_count += weight_length
                count += 1
                fss, extract_idx = extracting_fingerprints(embed_layers, self.local_fingerprints, self.extracting_matrices)
            
            if client_idx == 0:
                accuracy = test_model(new_model, self.main_task_test_loader)
            else:
                accuracy = accuracy
            logging.info("(Client_idx:{}, Result_idx:{}, FSS:{}, ACC:{}%)".format(client_idx, extract_idx, fss, accuracy))

            acc.append(accuracy)
            if client_idx == extract_idx:
                tvc.append(1)
            else:
                tvc.append(0)
            aggregated_models_dict[client_idx] = new_model
        

        # print('avg acc: %.2f%%' % (sum(acc)/len(acc)))
        # print('avg tvc: %.2f%%' % (sum(tvc)/len(tvc) * 100))


        # xxxxx

        if round % self.args.snap == 0:
            
            # logging.info("Training round %d" % round)
            # accuracy = test_model(new_model, self.main_task_test_loader)
            # logging.info("Main task accuracy: %.2f%%" % accuracy)
            accuracy = sum(acc)/len(acc)
            logging.info('Avg MA: %.2f%%' % accuracy)
            logging.info('Avg TR: %.2f%%' % (sum(tvc)/len(tvc) * 100))

        if self.best_acc < accuracy:
            self.best_acc = accuracy
        return aggregated_models_dict

    
    def agg_waffle(self, agent_models_dict, round):
        """ classic fed avg: Weighted average based on data size."""

        sm_updates, total_data = 0, 0
        # for _id, update in agent_updates_dict.items():
        #     n_agent_data = self.agent_data_sizes[_id]
        #     sm_updates +=  n_agent_data * update
        #     total_data += n_agent_data

        for agent_id, local_model in agent_models_dict.items():
            local_vector = parameters_to_vector(
                    [local_model.state_dict()[name] for name in local_model.state_dict()]
                ).detach()
            n_agent_data = self.agent_data_sizes[agent_id]
            sm_updates += local_vector * n_agent_data  # 累加所有 agent 的参数
            total_data += n_agent_data

        # 对 super_mask 部分的参数取均值
        # aggregated_vector = aggregated_vector / len(agent_models_dict)
        sm_updates /= total_data

        # Fine-tune new model here.
        new_model = copy.deepcopy(local_model)
        new_model = vector_to_model(sm_updates, new_model)

        pre_inject_acc = test_model(new_model, self.main_task_test_loader)

        new_model.train()
        optimizer = optim.SGD(new_model.parameters(), lr=self.learning_rate, momentum=0.0)


        for epoch in range(20):
            
            for inputs, targets in self.class_loaders['train']:
                
                inputs = inputs.cuda()
        #         # targets = torch.zeros(inputs.size(0), dtype=torch.long).cuda()  # 将所有 MNIST 图片标记为类 0
                targets = targets.cuda()
        #         # targets = _.cuda()
        #         optimizer.zero_grad()
                outputs = new_model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()

                optimizer.step()

            print(f"Epoch {epoch + 1}/20 finished.")
        
        # for i in range(10):
        #     # if i 
        #     acc = test_model(new_model, self.class_loaders['test'][i]['test'])
        #     # print(type(acc))
        #     # current_acc_trigger.append(builtins.round(acc, 2))

        #     print(acc)
        # xxxxx
        # new_model = copy.deepcopy(local_model)
        # new_model = vector_to_model(sm_updates, new_model)  # 假设你有一个 vector_to_model 函数

        # pre_inject_acc = 0
        aggregated_models_dict = {}
        for agent_id, local_model in agent_models_dict.items():
            # new_model = copy.deepcopy(local_model)
            # new_model = vector_to_model(sm_updates, new_model)  # 假设你有一个 vector_to_model 函数
            aggregated_models_dict[agent_id] = copy.deepcopy(new_model)
        
        if round % self.args.snap == 0:
            post_injection_acc = test_model(new_model, self.main_task_test_loader)
            logging.info(f"accuracy changes before and after injection: %.2f%% --> %.2f%%" % (pre_inject_acc, post_injection_acc))

            # logging.info("Training round %d" % round)
            # accuracy = test_model(new_model, self.main_task_test_loader)
            # logging.info("Main task accuracy: %.2f%%" % accuracy)

            if self.best_acc < post_injection_acc:
                self.best_acc = post_injection_acc
        return aggregated_models_dict


    def agg_tramark(self, agent_models_dict, cur_global_params, round):

        """
        1. Get the backdoor dataset
        2. Recover local model
        3. Perform individual backdoor injection to masked local model, save it
        4. Aggregate the masked part
        5. Return each personalized local model to local client.
        """

        if self.args.aggr == 'tramark_w_warmup' and round < int(self.args.rounds * self.args.warmup_rounds):
            return self.agg_avg(agent_models_dict, round)
        
        if self.args.aggr == 'tramark_w_warmup' and round == int(self.args.rounds * self.args.warmup_rounds):
            self.class_loaders = get_backdoor(self.args)
            self.super_mask, self.additional_mask = generate_masks_topk(ratio=0.99, model=list(agent_models_dict.values())[0])
            self.criterion = nn.CrossEntropyLoss()
            self.super_mask_flat = masks_to_vector(self.super_mask).cuda()
            self.additional_mask_flat = masks_to_vector(self.additional_mask).cuda()


        # 初始化一个聚合向量为零向量
        aggregated_vector = torch.zeros_like(self.super_mask_flat, dtype=torch.float32).cuda()
        # 遍历每个 agent 的模型，将其参数转为向量并加权累加
        total_data = 0
        for agent_id, local_model in agent_models_dict.items():
            local_vector = parameters_to_vector(
                    [local_model.state_dict()[name] for name in local_model.state_dict()]
                ).detach()
            n_agent_data = self.agent_data_sizes[agent_id]
            aggregated_vector += local_vector * n_agent_data  # 累加所有 agent 的参数
            total_data += n_agent_data

        # 对 super_mask 部分的参数取均值
        # aggregated_vector = aggregated_vector / len(agent_models_dict)
        aggregated_vector /= total_data

        # 创建一个新的字典，用于存储聚合后的模型
        aggregated_models_dict = {}

        # 遍历每个 agent，创建新的模型
        for agent_id, local_model in agent_models_dict.items():

            local_vector = parameters_to_vector(
                        [local_model.state_dict()[name] for name in local_model.state_dict()]
                    ).detach()

            # 根据掩码生成聚合后的参数向量
            final_vector = self.super_mask_flat * aggregated_vector + self.additional_mask_flat * local_vector

            # 创建新模型并加载聚合后的参数
            new_model = copy.deepcopy(local_model)
            new_model = vector_to_model(final_vector, new_model)  # 假设你有一个 vector_to_model 函数
            aggregated_models_dict[agent_id] = new_model

        main_task_acc = []

        # if round % 1 == 0:
        for id, local_model in aggregated_models_dict.items():
            if round % self.args.snap == 0:
                # logging.info("Training round %d" % round)
                # logging.info("Client %d, pre-injecting, Testing main task:" % id)
                pre_inject_acc = test_model(local_model, self.main_task_test_loader)
                # logging.info(f"Pre-injection accuracy: {pre_inject_acc:.2f}%")

            # print('Start to inject backdoor into local model with ID %d' % (id))
            optimizer = optim.SGD(local_model.parameters(), lr=0.0001, momentum=0.0)

            local_model.train()
            # bn_backup = save_bn_params(local_model)
            for epoch in range(5):
                
                for inputs, targets in self.class_loaders[id]['train']:
                    
                    inputs = inputs.cuda()
            #         # targets = torch.zeros(inputs.size(0), dtype=torch.long).cuda()  # 将所有 MNIST 图片标记为类 0
                    targets = targets.cuda()
            #         # targets = _.cuda()
            #         optimizer.zero_grad()
                    outputs = local_model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
            #         # model.apply_mask(task='additional', gradient=True)
                    for name, param in dict(local_model.named_parameters()).items():
                        if name in self.additional_mask:
                            if param.grad is not None:
                                param.grad.data.mul_(self.additional_mask[name].cuda())

                    optimizer.step()

                print(f"Epoch {epoch + 1}/5 finished.")
            if round % self.args.snap == 0:
                # logging.info("Client %d, post-injecting, Testing main task:" % id)
                # local_model = rewind_bn_params(bn_backup, local_model)
                post_injection_acc = test_model(local_model, self.main_task_test_loader)
                logging.info(f"Client %d, accuracy changes before and after injection: %.2f%% --> %.2f%%" % (id, pre_inject_acc, post_injection_acc))
                main_task_acc.append(post_injection_acc)
                # test_model(local_model, self.class_loaders[id]['test'])

            # local_model = unfreeze_bn(local_model)
            # local_model = rewind_bn_params(bn_backup, local_model)
            aggregated_models_dict[id] = copy.deepcopy(local_model)

        logging.info('-----'*4)

        verification = []
        if round % self.args.snap == 0:
            
            current_acc = sum(main_task_acc) / len(main_task_acc)
            # if round == 2:
            #     current_acc = -1
            # logging.info('Avg ACC: %.2f' % (current_acc))

            if self.args.data in ['tiny', 'fmnist'] or round >= self.args.rounds // 2:
                logging.info('--------Traceability verification--------')

                # best_acc_add_task = -1
                for id, local_model in aggregated_models_dict.items():
                    local_model.eval()
                    # print('-----'*4)

                    # print('Verify local model %d' % id)
                    current_acc_trigger = []
                    for i in range(10):
                        # if i 
                        acc = test_model(local_model, self.class_loaders[i]['test'])
                        # print(type(acc))
                        current_acc_trigger.append(builtins.round(acc, 2))

                    max_index = current_acc_trigger.index(max(current_acc_trigger))

                    if max_index == id:
                        verification.append(1)
                        logging.info('Client id: %d, accuracy on trigger set are %s, verficication success!' % (id, str(current_acc_trigger)))
                        # logging.info('Client id: %d, verficication success!' % id)
                    else:
                        verification.append(0)
                        logging.info('Client id: %d, accuracy on trigger set are %s, verficication fail!' % (id, str(current_acc_trigger)))
                    # logging.info('-----'*4)

                current_veri = sum(verification) / len(verification) * 100
                self.best_veri = current_veri


                # logging.info('Avg ACC: %.2f' % (current_acc))
                logging.info('Avg TVC: %.2f%%' % (current_veri))
                # else:
                    # logging.info('Avg ACC: %.2f' % (current_acc))

                if self.best_acc < current_acc:
                    self.best_acc = current_acc
                    # save_dict = {
                    #     "models": {model_id: model.state_dict() for model_id, model in aggregated_models_dict.items()}
                    # }

                    # # 保存到文件
                    # save_path = os.path.join(self.args.log_dir, "best_models.pt")
                    # torch.save(save_dict, save_path)

                # if sum(verification) == 10 and self.args.data == 'tiny':
                #     # 构造保存的字典
                #     save_dict = {
                #         "models": {model_id: model.state_dict() for model_id, model in aggregated_models_dict.items()}
                #     }

                #     # 保存到文件
                #     save_path = os.path.join(self.args.log_dir, "best_models_round_%d.pt" % round)
                #     torch.save(save_dict, save_path)
            # else:
            logging.info('Avg ACC: %.2f%%' % (current_acc))
                
        return aggregated_models_dict



    def agg_test(self, agent_models_dict, round):
        """ classic fed avg: Weighted average based on data size."""

        tvc = []
        if round != 1:
            for client_idx, local_model in agent_models_dict.items():
                # new_model = copy.deepcopy(local_model)
                # new_model = vector_to_model(sm_updates, new_model)  # 假设你有一个 vector_to_model 函数

                #local fingerprint insertion
                client_fingerprint = self.local_fingerprints[client_idx]
                embed_layers = get_embed_layers(local_model, self.name_dict[self.args.data])
                fss, extract_idx = extracting_fingerprints(embed_layers, self.local_fingerprints, self.extracting_matrices)
                logging.info("EVA: (Client_idx:{}, Result_idx:{}, FSS:{})".format(client_idx, extract_idx, fss))
                if client_idx == extract_idx:
                    tvc.append(1)
                else:
                    tvc.append(0)

            logging.info('Avg TR: %.2f%%' % (sum(tvc)/len(tvc) * 100))

        sm_updates, total_data = 0, 0
        # for _id, update in agent_updates_dict.items():
        #     n_agent_data = self.agent_data_sizes[_id]
        #     sm_updates +=  n_agent_data * update
        #     total_data += n_agent_data

        for agent_id, local_model in agent_models_dict.items():
            local_vector = parameters_to_vector(
                    [local_model.state_dict()[name] for name in local_model.state_dict()]
                ).detach()
            n_agent_data = self.agent_data_sizes[agent_id]
            sm_updates += local_vector * n_agent_data  # 累加所有 agent 的参数
            total_data += n_agent_data

        # 对 super_mask 部分的参数取均值
        # aggregated_vector = aggregated_vector / len(agent_models_dict)
        sm_updates /= total_data

        aggregated_models_dict = {}
        tvc = []
        acc = []
        for client_idx, local_model in agent_models_dict.items():
            new_model = copy.deepcopy(local_model)
            new_model = vector_to_model(sm_updates, new_model)  # 假设你有一个 vector_to_model 函数

            #local fingerprint insertion
            client_fingerprint = self.local_fingerprints[client_idx]
            embed_layers = get_embed_layers(new_model, self.name_dict[self.args.data])
            fss, extract_idx = extracting_fingerprints(embed_layers, self.local_fingerprints, self.extracting_matrices)

            count = 0
            while (extract_idx != client_idx or (client_idx == extract_idx and fss < 0.85))  and count <= 5:
                client_grad = calculate_local_grad(embed_layers,
                                                client_fingerprint,
                                                self.extracting_matrices[client_idx])
                client_grad = torch.mul(client_grad, -0.0005)
                weight_count = 0
                for embed_layer in embed_layers:
                    weight_length = embed_layer.weight.shape[0]
                    embed_layer.weight = torch.nn.Parameter(torch.add(embed_layer.weight, client_grad[weight_count: weight_count + weight_length].cuda())).cuda()
                    weight_count += weight_length
                count += 1
                fss, extract_idx = extracting_fingerprints(embed_layers, self.local_fingerprints, self.extracting_matrices)
            logging.info("AFT IJC: (Client_idx:{}, Result_idx:{}, FSS:{})".format(client_idx, extract_idx, fss))
            

            accuracy = test_model(new_model, self.main_task_test_loader)
            acc.append(accuracy)
            if client_idx == extract_idx:
                tvc.append(1)
            else:
                tvc.append(0)

            

            aggregated_models_dict[client_idx] = new_model
        

        # print('avg acc: %.2f%%' % (sum(acc)/len(acc)))
        # print('avg tvc: %.2f%%' % (sum(tvc)/len(tvc) * 100))


        # xxxxx

        if round % self.args.snap == 0:
            
            # logging.info("Training round %d" % round)
            # accuracy = test_model(new_model, self.main_task_test_loader)
            # logging.info("Main task accuracy: %.2f%%" % accuracy)
            accuracy = sum(acc)/len(acc)
            logging.info('Avg MA: %.2f%%' % accuracy)
            logging.info('Avg TR: %.2f%%' % (sum(tvc)/len(tvc) * 100))

        if self.best_acc < accuracy:
            self.best_acc = accuracy
        return aggregated_models_dict
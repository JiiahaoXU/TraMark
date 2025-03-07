from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torch
import random
import torch.nn as nn
import logging


def create_class_specific_datasets(dataset, num_train_per_class=None, transform=None):
    """
    为每个类别创建指定数量的训练样本和所有测试样本的数据集。
    
    参数:
    - dataset: 原始数据集对象。
    - num_train_per_class: 每个类别的训练样本数量，如果为 None，则提取所有样本。
    - transform: 数据集的预处理方式。
    
    返回:
    - class_specific_datasets: 每个类别的训练和测试数据集，字典形式。
    """
    class_specific_datasets = {}
    class_indices = {label: [] for label in range(10)}

    # 收集每个类别的索引
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # 创建每个类别的数据子集
    for label in range(10):
        if num_train_per_class is None:
            # 提取所有样本
            selected_indices = class_indices[label]
        else:
            # 提取前 num_train_per_class 张
            selected_indices = class_indices[label][:num_train_per_class]

        class_specific_datasets[label] = Subset(dataset, selected_indices)

    return class_specific_datasets


def create_class_specific_loaders(train_dataset, test_dataset, num_train_per_class, batch_size=32, num_workers=2):
    """
    创建每个类别的训练和测试数据加载器。
    
    参数:
    - train_dataset: 原始训练数据集。
    - test_dataset: 原始测试数据集。
    - num_train_per_class: 每个类别的训练样本数量。
    - batch_size: 数据加载器的批量大小。
    - num_workers: 数据加载器的线程数。
    
    返回:
    - loaders: 每个类别的训练和测试加载器，字典形式。
    """

    # if random_train:
    #     train_indices = random.sample(range(len(train_dataset)), 1000)  # 随机选择 1000 张训练图片
    #     train_dataset = Subset(train_dataset, train_indices)
    #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    train_class_datasets = create_class_specific_datasets(train_dataset, num_train_per_class)
    test_class_datasets = create_class_specific_datasets(test_dataset, num_train_per_class=None)  # 提取所有测试样本

    loaders = {
        label: {
            'train': DataLoader(train_class_datasets[label], batch_size=batch_size, shuffle=True, num_workers=num_workers),
            'test': DataLoader(test_class_datasets[label], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        }
        for label in range(10)
    }

    return loaders


def get_random_subset(dataset, num_samples=1000):
    """
    随机选择指定数量的样本创建数据子集。
    
    参数:
    - dataset: 原始数据集。
    - num_samples: 需要选择的样本数量。
    
    返回:
    - subset: 由随机选择的样本组成的数据子集。
    """
    selected_indices = random.sample(range(len(dataset)), num_samples)
    return Subset(dataset, selected_indices)


def get_backdoor(args):
    # 数据预处理

    if args.data == 'tiny':
        transform_mnist = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(64),
        # transforms.Resize(28),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # transforms.Normalize(0.5, 0.5)
        ])
    elif args.data == 'fmnist':
        transform_mnist = transforms.Compose([
        # transforms.Grayscale(3),
        transforms.Resize(28),
        # transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
        ])
    elif args.data == 'cifar10':
        transform_mnist = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(32),
        # transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    elif args.data == 'cifar100':
        transform_mnist = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(32),
        # transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])


    # 加载 MNIST 数据集
    train_mnist = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform_mnist)
    test_mnist = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform_mnist)

    
    
    if args.aggr == 'waffle':
        train_mnist = get_random_subset(train_mnist, num_samples=1000)
        class_loaders = {
            'train': DataLoader(train_mnist, batch_size=32, shuffle=True, num_workers=args.num_workers),
            'test': create_class_specific_loaders(train_mnist, test_mnist, num_train_per_class=100, batch_size=32, num_workers=args.num_workers)
        }
    else:
        class_loaders = create_class_specific_loaders(train_mnist, test_mnist, num_train_per_class=100, batch_size=32, num_workers=args.num_workers)
    

    return class_loaders


def generate_masks(ratio, model):
    
    keys = [name for name in model.state_dict()]
    # print(keys)
    keys_parameters = [name for name, param in model.named_parameters()]
    # print(keys_parameters)

    # xx
    param_dict = {name: param for name, param in model.named_parameters()}
    super_mask = {}
    additional_mask = {}


    for name in keys:
        if name in keys_parameters:
        # for name, param in param_dict.items():
            param = param_dict[name]
            numel = param.numel()
            mask = torch.zeros(numel, dtype=torch.bool)
        # if name in keys_parameters:
            selected_indices = random.sample(range(numel), int(numel * ratio))
        else:
            param = model.state_dict()[name]
            numel = param.numel()
            mask = torch.zeros(numel, dtype=torch.bool)
            selected_indices = random.sample(range(numel), int(numel * 1.0))
        mask[selected_indices] = True

        super_mask[name] = mask.view_as(param)
        additional_mask[name] = ~mask.view_as(param)  # 互补掩码
        # else:


    # print(super_mask[name])
    # print(additional_mask[name])
    return super_mask, additional_mask


# import torch

def generate_masks_topk(ratio, model):
    keys = [name for name in model.state_dict()]
    keys_parameters = [name for name, param in model.named_parameters()]
    
    param_dict = {name: param for name, param in model.named_parameters()}
    super_mask = {}
    additional_mask = {}

    for name in keys:
        if name in keys_parameters:  # Directly process all parameters
            param = param_dict[name]
            numel = param.numel()
            mask = torch.zeros(numel, dtype=torch.bool)

            # Compute absolute values and select top-k elements
            flat_param = param.view(-1).abs()
            topk = int(numel * ratio)
            if topk > 0:
                _, selected_indices = torch.topk(flat_param, topk, largest=True)
                mask[selected_indices] = True  # Mark top-k parameters

        else:
            param = model.state_dict()[name]
            numel = param.numel()
            mask = torch.ones(numel, dtype=torch.bool)  # Other parameters fully belong to super_mask

        super_mask[name] = mask.view_as(param)
        additional_mask[name] = ~mask.view_as(param)  # Complementary mask

    return super_mask, additional_mask



def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            module.requires_grad_(False)
    return model

def unfreeze_bn(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.train()
            module.requires_grad_(True)
    return model

def masks_to_vector(masks):
    """
    将掩码字典展平为单个向量。
    
    参数:
        masks (dict): 一个包含掩码的字典。
    
    返回:
        torch.Tensor: 所有掩码展平并拼接后的向量。
    """
    vectors = []
    for name, mask in masks.items():
        # if name == 'bn1.weight':
        #     print(mask)
        vectors.append(mask.flatten())  # 展平每个掩码
    # xxx
    return torch.cat(vectors)  # 拼接为一个向量


def test_model(model, dataloader):
    """Test the model with the given dataloader and mask."""
    model.eval()
    # if task != 'super':
    #     model.save_current_params()
    #     model.apply_mask(task=task, gradient=False)

    total, correct = 0, 0
    with torch.no_grad():
        # for inputs, targets in dataloader:
        #     inputs, targets = inputs.cuda(), targets.cuda()
        #     outputs = model(inputs)
        #     _, predicted = outputs.max(1)
        #     total += targets.size(0)
        #     correct += predicted.eq(targets).sum().item()
        # elif task == 'additional':
        for inputs, targets in dataloader:
            inputs = inputs.cuda()
            # targets = torch.zeros(inputs.size(0), dtype=torch.long).cuda()  # 将所有图片标记为类 0
            targets = targets.cuda()  # 将所有图片标记为类 0

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    accuracy = 100. * correct / total
    # logging.info(f"Accuracy: {accuracy:.2f}%")
    # if task != 'super':
    #     model.rewind_params(task=task)

    return accuracy
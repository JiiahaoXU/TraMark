from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torch
import random
import torch.nn as nn
import logging


def create_class_specific_datasets(dataset, num_train_per_class=None, transform=None):

    class_specific_datasets = {}
    class_indices = {label: [] for label in range(10)}

    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    for label in range(10):
        if num_train_per_class is None:
            selected_indices = class_indices[label]
        else:
            selected_indices = class_indices[label][:num_train_per_class]

        class_specific_datasets[label] = Subset(dataset, selected_indices)

    return class_specific_datasets


def create_class_specific_loaders(train_dataset, test_dataset, num_train_per_class, batch_size=32, num_workers=2):
    
    train_class_datasets = create_class_specific_datasets(train_dataset, num_train_per_class)
    test_class_datasets = create_class_specific_datasets(test_dataset, num_train_per_class=None) 

    loaders = {
        label: {
            'train': DataLoader(train_class_datasets[label], batch_size=batch_size, shuffle=True, num_workers=num_workers),
            'test': DataLoader(test_class_datasets[label], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        }
        for label in range(10)
    }

    return loaders


def get_watermark_dataset(args):

    if args.data == 'tiny':
        transform_mnist = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])
    elif args.data == 'fmnist':
        transform_mnist = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
        ])
    elif args.data == 'cifar10':
        transform_mnist = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    elif args.data == 'cifar100':
        transform_mnist = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])


    train_mnist = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform_mnist)
    test_mnist = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform_mnist)
    
    class_loaders = create_class_specific_loaders(train_mnist, test_mnist, num_train_per_class=100, batch_size=32, num_workers=args.num_workers)
    

    return class_loaders


def generate_masks_topk(partition_ratio, model):

    partition_ratio = 1 - partition_ratio
    keys = [name for name in model.state_dict()]
    keys_parameters = [name for name, param in model.named_parameters()]
    
    param_dict = {name: param for name, param in model.named_parameters()}
    main_task_mask = {}
    watermarking_mask = {}

    for name in keys:
        if name in keys_parameters:  
            param = param_dict[name]
            numel = param.numel()
            mask = torch.zeros(numel, dtype=torch.bool)

            flat_param = param.view(-1).abs()
            topk = int(numel * partition_ratio)
            if topk > 0:
                _, selected_indices = torch.topk(flat_param, topk, largest=True)
                mask[selected_indices] = True 

        else:
            param = model.state_dict()[name]
            numel = param.numel()
            mask = torch.ones(numel, dtype=torch.bool)  

        main_task_mask[name] = mask.view_as(param)
        watermarking_mask[name] = ~mask.view_as(param)  

    return main_task_mask, watermarking_mask


def masks_to_vector(masks):

    vectors = []
    for name, mask in masks.items():
        vectors.append(mask.flatten())  

    return torch.cat(vectors)  


def test_model(model, dataloader):
    """Test the model with the given dataloader."""
    model.eval()
    
    total, correct = 0, 0
    with torch.no_grad():
    
        for inputs, targets in dataloader:
            inputs = inputs.cuda()
            targets = targets.cuda()  

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    accuracy = 100. * correct / total

    return accuracy
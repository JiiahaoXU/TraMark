## Traceable Black-box Watermarks for Federated Learning

We provide the code of proposed TraMark.

## Usage

### Environment

Our code does not rely on special libraries or tools, so it can be easily integrated with most environment settings. 

If you want to use the same settings as us, we provide the conda environment we used in `env.yaml` for your convenience.

### Dataset

All tested datasets are available on `torchvision` and will be downloaded automatically, except for Tiny-ImageNet, which can be easily downloaded from Kaggle.

### Example

Generally, to run a case with default settings, you can easily use the following command:

```
python federated.py --aggr tramark --data cifar10 --alpha 0.5 --k 0.01
```

If you want to run a case with non-IID settings, you can easily use the following command:

```
python federated.py --aggr tramark --data cifar10 --alpha 0.5 --k 0.01 --non_iid --gamma 0.5

```

Here,

| Argument        | Type       | Description   | Choice |
|-----------------|------------|---------------|--------|
| `aggr`         | str   | Watermarking method | avg, tramark |
| `data`    |   str     | Main task dataset          | fmnist, cifar10, cifar100, tiny |
| `non_iid`         | store_true | Enable non-IID settings or not      | N/A |
| `gamma`         | float | Data heterogeneous degree     | from 0.1 to 1.0|

For other arguments, you can check the `federated.py` file where the detailed explanation is presented.



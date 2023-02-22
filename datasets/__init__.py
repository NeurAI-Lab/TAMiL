from datasets.seq_cifar100 import SequentialCIFAR100
from datasets.seq_cifar10 import SequentialCIFAR10
from datasets.seq_core50 import SequentialCore50
from datasets.seq_tinyimagenet import SequentialTinyImagenet
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace

NAMES = {
    SequentialCIFAR100.NAME: SequentialCIFAR100,
    SequentialCIFAR10.NAME: SequentialCIFAR10,
    SequentialCore50.NAME: SequentialCore50,
    SequentialTinyImagenet.NAME: SequentialTinyImagenet,
}

GCL_NAMES = {
}


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)


def get_gcl_dataset(args: Namespace):
    """
    Creates and returns a GCL dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in GCL_NAMES.keys()
    return GCL_NAMES[args.dataset](args)

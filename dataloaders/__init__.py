from datasets import ML1MDataset
from .bert import BertDataloader

__all__ = ['ML1MDataset', 'BertDataloader']

DATALOADERS = {BertDataloader.code(): BertDataloader}


def dataloader_factory(args):
    dataset = ML1MDataset(args)
    dataloader = BertDataloader(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()
    return train, val, test

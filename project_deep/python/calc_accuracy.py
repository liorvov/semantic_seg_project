import torch


def calc_accuracy(label, output):
    return torch.sum(torch.eq(label, output)) / torch.sum(label != -1)

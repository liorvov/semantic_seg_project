from camvidPixelLabelIDs import *
from countEachLabel import *
from partitionCamVidData import *
import pandas as pd
import os
import requests
import zipfile
import torch
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch import nn
from PIL import Image
import cv2 as cv


def reverse_label_ids(label_ids):
    ret = {}
    for id in label_ids:
        for color in label_ids[id]:
            ret[tuple(color)] = id
    return ret


def label_to_colormap(label, batch_size, reversed_label_ids, classes):
    colormap = torch.mul(torch.ones((batch_size, 720, 960)), -1)
    for batch in range(len(colormap)):
        for key in reversed_label_ids:
            tmp = cv.inRange(np.asarray(label[batch].data.cpu()), np.array(key), np.array(key))
            colormap[batch][tmp > 0] = classes.index(reversed_label_ids[key])
    return colormap


def label_to_reduced(label, reversed_label_ids, label_ids):
    reduced = torch.empty((720, 960, 3))
    for i in range(len(label)):
        for j in range(len(label[0])):
            color = tuple(label[i][j])
            if color == (0, 0, 0):
                reduced[i][j] = torch.tensor([0, 0, 0])
            else:
                reduced[i][j] = torch.tensor(label_ids[reversed_label_ids[color]][0])
            for k in reduced[i][j]:
                if k < 0 or k > 255:
                    print(reduced[i][j])
    return reduced.int()

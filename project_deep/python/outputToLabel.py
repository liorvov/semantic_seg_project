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


def output_to_label(out, label_ids_list_r, label_ids_list_g, label_ids_list_b, batch_size):
    label = torch.empty((3, batch_size, 720, 960))
    indexes = torch.argmax(out, dim=3).detach().cpu()
    label[0] = torch.take(torch.tensor(label_ids_list_r), indexes)
    label[1] = torch.take(torch.tensor(label_ids_list_g), indexes)
    label[2] = torch.take(torch.tensor(label_ids_list_b), indexes)
    label = label.permute(1, 2, 3, 0)
    return label


def validOutputToLabel(out, classes):
    label = torch.empty((720, 960, 3))
    labelIDs = camvidPixelLabelIDs()
    for row in range(len(label)):
        for col in range(len(label[0])):
            index = out[row][col].argmax()
            label[row][col] = torch.from_numpy(labelIDs[classes[index]][0])
    return label
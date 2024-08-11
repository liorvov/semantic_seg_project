from camvidPixelLabelIDs import *
from countEachLabel import *
from partitionCamVidData import *
from zigzag import *
from labelToColormap import *
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
import tensorflow as tf


class LSTMDataset(Dataset):
    def __init__(self, model, dataloader, use_resent50, batch_size, num_classes, reversed_label_ids, classes):
        print("setting LSTM dataset...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
        self.lstm_inputs = []
        self.lstm_labels = []
        n = 0
        for img, lbl in dataloader:
            img = img.permute(0, 3, 2, 1).float().to(device)
            with torch.no_grad():
                if use_resent50:
                    output = model(img)["out"].permute(0, 3, 2, 1).float().to(torch.device("cpu"))
                else:
                    output = model(img).permute(0, 3, 2, 1).float().to(torch.device("cpu"))
            label_colormap = label_to_colormap(lbl, batch_size, reversed_label_ids, classes)
            for i in range(batch_size):
                print(n)
                n = n + 1
                lstm_input = zigzagify(output[i], num_classes)
                lstm_label = zigzagify(label_colormap[i], 1).squeeze(1)
                self.lstm_inputs.append(lstm_input)
                self.lstm_labels.append(lstm_label)

    def __len__(self):
        return len(self.lstm_labels)

    def __getitem__(self, idx):
        lstm_label = self.lstm_labels[idx]
        lstm_input = self.lstm_inputs[idx]
        return lstm_input, lstm_label

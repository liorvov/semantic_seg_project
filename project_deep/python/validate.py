from camvidPixelLabelIDs import *
from countEachLabel import *
from partitionCamVidData import *
from labelToColormap import *
from outputToLabel import *
from calc_accuracy import *

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
import seaborn as sn
from sklearn.metrics import confusion_matrix


def validate(model, dataloader, batch_size, reversedLabelIDs, classes, labelIDsListR, labelIDsListG, labelIDsListB,
             numOfImages, epoch, writer, name, use_resent50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy = 0
    batch_counter = 0
    cf = np.zeros((11, 11))
    for img, lbl in dataloader:
        torch.cuda.empty_cache()
        img = img.permute(0, 3, 2, 1).float()
        img = img.to(device)
        lbl = label_to_colormap(lbl, batch_size, reversedLabelIDs, classes).to(device)
        with torch.no_grad():
            out = model(img)["out"] if use_resent50 else model(img)
        output = out.permute(0, 3, 2, 1).float()
        output = label_to_colormap(output_to_label(output, labelIDsListR, labelIDsListG, labelIDsListB, batch_size),
                                   batch_size, reversedLabelIDs, classes).to(device)
        if name == "test":
            pred_flat = list(np.concatenate(torch.max(out, 1).indices.data.cpu().int().tolist()).flat)
            label_flat = list(np.concatenate(lbl.data.permute(0, 2, 1).cpu().int().tolist()).flat)
            for i in range(len(pred_flat)):
                if label_flat[i] != -1:
                    cf[label_flat[i]][pred_flat[i]] += 1
            del pred_flat
            del label_flat
        accuracy += calc_accuracy(lbl, output)
        batch_counter += 1
        print("{}: {} out of {}".format(name, batch_counter * batch_size, numOfImages))
        del img
        del lbl
        del out
        del output
    accuracy = accuracy * 100 / batch_counter
    writer.add_scalar('{} accuracy'.format(name), accuracy, epoch)
    print("{} accuracy = {:.3f}%".format(name, accuracy))

    if name == "test":
        print("generating confusion matrix...")
        cf = cf / np.sum(cf, axis=1)[:, None]
        plt.figure(figsize=(12, 7))
        sn.heatmap(pd.DataFrame(cf / np.sum(cf, axis=1)[:, None], index=[i for i in classes],
                   columns=[i for i in classes]), annot=True)


def validate_lstm(model, lstm_model, dataloader, batch_size, reversedLabelIDs, classes, labelIDsListR, labelIDsListG,
                  labelIDsListB,  numOfImages, epoch, writer, name, use_resent50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy = 0
    batch_counter = 0
    y_pred, y_true = [], []
    cf = np.zeros((11, 11))
    for img, lbl in dataloader:
        torch.cuda.empty_cache()
        lbl = lbl.long().to(device)
        for i in range(batch_size):
            lstm_input = img[i].unsqueeze(0).to(device)
            with torch.no_grad():
                lstm_out = lstm_model(lstm_input)
            if name == "test":
                pred_flat = list(np.concatenate(torch.max(lstm_out, 2).indices.data.cpu().int().tolist()).flat)
                label_flat = list(np.concatenate(lbl[i].unsqueeze(0).data.cpu().int().tolist()).flat)
                for j in range(len(pred_flat)):
                    if label_flat[j] != -1:
                        cf[label_flat[j]][pred_flat[j]] += 1
                del pred_flat
                del label_flat
            lstm_result = torch.argmax(lstm_out.squeeze(0), dim=1)
            accuracy += calc_accuracy(lbl[i], lstm_result) * 100 / numOfImages
            del lstm_input
            del lstm_out
            del lstm_result
        del img
        del lbl
        batch_counter += 1
        print(f"lstm {name}: {batch_counter * batch_size} out of {numOfImages}")
    writer.add_scalar(f'lstm {name} accuracy', accuracy, epoch)
    print(f"lstm {name} accuracy = {accuracy:.3f}%")

    if name == "test":
        print("generating lstm confusion matrix...")
        cf = cf / np.sum(cf, axis=1)[:, None]
        plt.figure(figsize=(12, 7))
        sn.heatmap(pd.DataFrame(cf / np.sum(cf, axis=1)[:, None], index=[i for i in classes],
                   columns=[i for i in classes]), annot=True)

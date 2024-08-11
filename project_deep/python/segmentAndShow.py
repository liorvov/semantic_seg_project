from camvidPixelLabelIDs import *
from countEachLabel import *
from partitionCamVidData import *
from outputToLabel import *
from labelToColormap import *
from validate import *
from calc_accuracy import *
from torch.utils.data import Dataset
from zigzag import *
import os
import requests
import zipfile
import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


def segment_and_show(image, label, model, lstm_model, label_ids_list_r, label_ids_list_g, label_ids_list_b,
                     use_resent50, reversed_label_ids, classes, label_ids, length, width):
    print("generating random test image segmentation...")
    img_t = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 2, 1).float().to('cuda')
    label = label_to_reduced(label, reversed_label_ids, label_ids)
    model.eval()
    lstm_model.eval()
    out = model(img_t)["out"] if use_resent50 else model(img_t)
    output = out.permute(0, 3, 2, 1).float().to(torch.device("cpu"))
    model_pred = output_to_label(output, label_ids_list_r, label_ids_list_g, label_ids_list_b, 2)
    model_pred_over_image = (model_pred[0] * 0.4 + image * 0.6) / 256
    label_over_image = (label * 0.4 + image * 0.6) / 256
    fig = plt.figure(figsize=(10, 7))
    black = torch.mul(torch.ones((1, 720, 960)), -1)
    label_colormap = label_to_colormap(label.unsqueeze(0), 1, reversed_label_ids, classes)
    model_colormap = label_to_colormap(model_pred, 1, reversed_label_ids, classes)
    model_diff = torch.bitwise_not(torch.bitwise_or(torch.eq(model_colormap, label_colormap),
                                                    torch.eq(label_colormap, black))).squeeze()
    lstm_input = zigzagify(output[0], len(classes)).unsqueeze(0).to(torch.device("cuda"))
    lstm_output = lstm_model(lstm_input).squeeze(0)
    lstm_reshaped = dezigzagify(lstm_output, len(classes), length, width).unsqueeze(0)
    lstm_pred = output_to_label(lstm_reshaped, label_ids_list_r, label_ids_list_g, label_ids_list_b, 2)
    lstm_colormap = label_to_colormap(lstm_pred, 1, reversed_label_ids, classes)
    lstm_model_diff = torch.bitwise_not(torch.eq(model_colormap, lstm_colormap)).squeeze()
    lstm_orig_diff = torch.bitwise_not(torch.bitwise_or(torch.eq(lstm_colormap, label_colormap),
                                                        torch.eq(label_colormap, black))).squeeze()
    lstm_pred_over_image = (lstm_pred[0] * 0.4 + image * 0.6) / 256

    fig.add_subplot(2, 5, 1)
    plt.imshow(image)
    plt.title("original image")

    fig.add_subplot(2, 5, 2)
    plt.imshow(model_diff, cmap='gray', vmin=0, vmax=1)
    plt.title("model labels diff from original")

    fig.add_subplot(2, 5, 3)
    plt.imshow(label)
    plt.title("original labels")

    fig.add_subplot(2, 5, 4)
    plt.imshow((model_pred[0]) / 256)
    plt.title("model labels")

    fig.add_subplot(2, 5, 5)
    plt.imshow((lstm_pred[0]) / 256)
    plt.title("lstm model labels")

    fig.add_subplot(2, 5, 6)
    plt.imshow(lstm_model_diff, cmap='gray', vmin=0, vmax=1)
    plt.title("lstm labels diff from model")

    fig.add_subplot(2, 5, 7)
    plt.imshow(lstm_orig_diff, cmap='gray', vmin=0, vmax=1)
    plt.title("lstm labels diff from original")

    fig.add_subplot(2, 5, 8)
    plt.imshow(label_over_image)
    plt.title("original labels overlap")

    fig.add_subplot(2, 5, 9)
    plt.imshow(model_pred_over_image)
    plt.title("model labels overlap")

    fig.add_subplot(2, 5, 10)
    plt.imshow(lstm_pred_over_image)
    plt.title("lstm model labels overlap")

    plt.show()

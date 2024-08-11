import numpy as np
import torch


def zigzagify(mat, classes):
    length = len(mat)
    width = len(mat[0])
    arr = torch.zeros((length * width, classes))
    t = 0
    for i in range(length + width - 1):
        if i % 2:
            x = 0 if i < width else i - width + 1
            y = i if i < width else width - 1
            while x < length and y >= 0:
                arr[t] = mat[x][y]
                t = t + 1
                x = x + 1
                y = y - 1
        else:
            x = i if i < length else length - 1
            y = 0 if i < length else i - length + 1
            while x >= 0 and y < width:
                arr[t] = mat[x][y]
                t = t + 1
                x = x - 1
                y = y + 1
    return arr


def dezigzagify(arr, classes, length, width):
    mat = torch.zeros((length, width, classes))
    t = 0
    for i in range(length + width - 1):
        if i % 2:
            x = 0 if i < width else i - width + 1
            y = i if i < width else width - 1
            while x < length and y >= 0:
                mat[x][y] = arr[t]
                t = t + 1
                x = x + 1
                y = y - 1
        else:
            x = i if i < length else length - 1
            y = 0 if i < length else i - length + 1
            while x >= 0 and y < width:
                mat[x][y] = arr[t]
                t = t + 1
                x = x - 1
                y = y + 1
    return mat

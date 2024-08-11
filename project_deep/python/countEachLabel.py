import numpy as np
import cv2 as cv
import pandas as pd

def count_each_label(data, label_ids):
    tbl = pd.DataFrame(index=label_ids)
    tbl["PixelCount"] = np.zeros(len(label_ids)).astype(int)
    tbl["ImagePixelCount"] = np.zeros(len(label_ids)).astype(int)
    label0 = data[0][1]
    width, height = len(label0), len(label0[0])
    image_size = width * height
    for i in range(len(data)):
        lbl = data[i][1]
        prev_image_pixel_count = tbl["ImagePixelCount"].copy()
        for label, colors in label_ids.items():
            for color in colors:
                tmp = cv.inRange(lbl, np.array(color), np.array(color))
                count = cv.countNonZero(tmp)
                tbl["PixelCount"][label] += count
                tbl["ImagePixelCount"][label] = max(tbl["ImagePixelCount"][label], prev_image_pixel_count[label] +
                                                    image_size * (count != 0))
    return tbl

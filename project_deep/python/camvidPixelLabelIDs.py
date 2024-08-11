import numpy as np

# Return the label IDs corresponding to each class.
#
# The CamVid dataset has 32 classes. Group them into 11 classes following
# the original SegNet training methodology.
#
# The 11 classes are:
#
#  "Sky" "Building", "Pole", "Road", "Pavement", "Tree", "SignSymbol",
#  "Fence", "Car", "Pedestrian",  and "Bicyclist".
#
# CamVid pixel label IDs are provided as RGB color values. Group them into
# 11 classes and return them as a cell array of M-by-3 matrices. The
# original CamVid class names are listed alongside each RGB value. Note
# that the Other/Void class are excluded below.


def camvid_pixel_label_ids():

    label_ids = {
        "Sky":      [np.array([128, 128, 128])],
        "Building": [np.array([64, 128, 0]),
                     np.array([0, 0, 128]),
                     np.array([0, 192, 64]),
                     np.array([64, 0, 64]),
                     np.array([128, 0, 192])
                     ],
        "Pole":     [np.array([128, 192, 192]),
                     np.array([64, 0, 0])
                     ],
        "Road":     [np.array([128, 64, 128]),
                     np.array([192, 0, 128]),
                     np.array([64, 0, 192])
                     ],
        "Pavement": [np.array([192, 0, 0]),
                     np.array([128, 192, 64]),
                     np.array([192, 128, 128])
                     ],
        "Tree":     [np.array([0, 128, 128]),
                     np.array([0, 192, 192])
                     ],
        "SignSymbol": [np.array([128, 128, 192]),
                       np.array([64, 128, 128]),
                       np.array([64, 64, 0])
                       ],
        "Fence":    [np.array([128, 64, 64])
                     ],
        "Car":      [np.array([128, 0, 64]),
                     np.array([192, 128, 64]),
                     np.array([192, 128, 192]),
                     np.array([128, 64, 192]),
                     np.array([64, 64, 128])
                     ],
        "Pedestrian":   [np.array([0, 64, 64]),
                         np.array([64, 128, 192]),
                         np.array([192, 0, 64]),
                         np.array([64, 128, 64])
                         ],
        "Bicyclist":    [np.array([192, 128, 0]),
                         np.array([192, 0, 192])
                         ]
    }

    return label_ids

import numpy as np
import pandas as pd
import scipy.io as scio
import torch
from torch import nn

def Compute_TP_TN_FP_FN(test_label, label, matrix):
    for i in range(len(test_label)):
        if test_label[i] == 0 and label[i] == 0:  # TP
            matrix[0] += 1
        elif test_label[i] == 1 and label[i] == 1:  # TN
            matrix[1] += 1
        elif test_label[i] == 1 and label[i] == 0:  # FP
            matrix[2] += 1
        elif test_label[i] == 0 and label[i] == 1:  # FN
            matrix[3] += 1
    return matrix

# init weight
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

# get datas and labels
def GetData(filename, lablepath):
    tmp = scio.loadmat(filename)
    xdata = tmp['EEG']
    xdata = xdata.reshape(20, 31, 5, 168)
    xdata_normal = xdata / np.linalg.norm(xdata, axis=3, keepdims=True)
    band = np.zeros(shape=(5, 19, 31, 168))

    for i in range(5):
        band[i] = xdata_normal[1:20, :, i, :].reshape(19, 31, 168)

    tmplable = pd.read_excel(lablepath)
    tmplable = tmplable.iloc[1:20]
    subindex = np.array(tmplable.iloc[:, 1]) - 2
    sublabel = np.array(tmplable.iloc[:, 4])

    sublabel = np.array(train_lable(sublabel))
    labelLen = sublabel.shape[0]
    ydata = np.zeros(labelLen, dtype=np.longlong)

    for i in range(labelLen):
        ydata[i] = sublabel[i]

    return band, ydata, subindex

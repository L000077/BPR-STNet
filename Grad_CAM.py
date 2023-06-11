import mne
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn
from matplotlib import gridspec
from sklearn.metrics import accuracy_score
from torch import optim, nn

import BPR-STNet
from Utli import GetData, init_weights
from grad_cam.utils import GradCAM

# Grad-CAM
def grad_cam(i, xdata, model, target_layers, probs, true_label):
    input_tensor = xdata

    n1 = probs[0]
    target_category = int(np.argmax(n1))

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    max_cam = np.argmax(grayscale_cam) % 168

    cam_avg = input_tensor.cpu().numpy()
    cam_avg = np.sum(np.multiply(cam_avg[0], grayscale_cam), axis=2)

    max_gamma = xdata[:, :, :, max_cam]
    max_gamma = max_gamma.cpu().numpy()

    cam_avg = (cam_avg - np.min(cam_avg)) / (np.max(cam_avg) - np.min(cam_avg))

    fig = plt.figure(figsize=(10, 4))
    gridlayout = gridspec.GridSpec(nrows=4, ncols=6, figure=fig, wspace=-0.1, hspace=0.1, top=0.9)
    ax1 = fig.add_subplot(gridlayout[0:4, 0:4])
    ax2 = fig.add_subplot(gridlayout[0:2, 4:6])
    ax3 = fig.add_subplot(gridlayout[2:4, 4:6])

    montage = mne.channels.read_custom_montage('31.locs')
    info = mne.create_info(ch_names=montage.ch_names, sfreq=128, ch_types='eeg')

    evoked1 = mne.EvokedArray(cam_avg.reshape(31, 1), info)
    evoked1.set_montage(montage)
    evoked2 = mne.EvokedArray(max_gamma.reshape(31, 1), info)
    evoked2.set_montage(montage)
    im1, cm = mne.viz.plot_topomap(evoked1.data[:, 0], evoked1.info, axes=ax2, show=False, names=montage.ch_names,
                                   show_names=True, cmap='coolwarm', vmin=0, vmax=1)
    im2, cm = mne.viz.plot_topomap(evoked2.data[:, 0], evoked2.info, axes=ax3, show=False, names=montage.ch_names,
                                   show_names=True, cmap='coolwarm',vmin=0, vmax=0.15)
    plt.colorbar(im1, ax=ax2)
    plt.colorbar(im2, ax=ax3)

    seaborn.heatmap(grayscale_cam[0], ax=ax1, cmap="coolwarm")

    fig.suptitle('Subject:' + str(i + 1) + '    True Label:' + str(true_label) + '   P(Novice)=' + str(
        round(n1[0], 3)) + '   P(Expert)=' + str(round(n1[1], 3)))

    fig_name = 'figure/theta_subject'+str(i)+'.svg'
    plt.savefig(fig_name)

def run():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    filename = r'data/20_powerDataset.mat'
    lablepath = 'E:\LOLpro\lable.xlsx'

    tmplabel = pd.read_excel(lablepath)
    tmplabel = tmplabel.iloc[1:20]
    sublabel = np.array(tmplabel.iloc[:, 4])
    band, ydata, subindex = GetData(filename, lablepath)
    tempData = band[4, :, :, :]
    tempData = tempData.reshape(19, 1, 31, 168)

    channel = 31
    subject = 19
    lr = 1e-3
    batch_size = 4
    n_epoch = 200

    result = np.zeros(subject)

    for i in range(0, 19):

        trainIndex = np.where(subindex != i)[0]
        xTrain = tempData[trainIndex]
        testIndex = np.where(subindex == i)[0]
        xTest = tempData[testIndex]

        x_train = xTrain.reshape(18, 1, channel, 168)
        y_train = ydata[trainIndex]

        x_test = xTest.reshape(1, 1, channel, 168)
        y_test = ydata[testIndex]

        train = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

        model = myNet.net().to(device)
        model.apply(init_weights)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch, eta_min=0, last_epoch=-1)
        loss_class = nn.CrossEntropyLoss().cuda()

        for p in model.parameters():
            p.requires_grad = True

        for epoch in range(n_epoch):
            for j, data in enumerate(train_loader, 0):
                inputs, labels = data
                input_data = inputs.to(device)
                class_label = labels.to(device)
                model.zero_grad()
                model.train()
                class_output = model(input_data)
                err_label = loss_class(class_output, class_label)
                err = err_label
                err.backward()
                optimizer.step()
                scheduler.step()

        # test the results
        model.train(False)
        with torch.no_grad():
            x_test = torch.Tensor(x_test).cuda()
            answer = model(x_test)
            probs = answer.cpu().numpy()
            preds = probs.argmax(axis=-1)
            acc = accuracy_score(y_test, preds)

            print('SoftMax:', probs)
            result[i] = acc

        target_layers = [model.layer2]
        grad_cam(i, x_test, model, target_layers, probs, sublabel[i])

if __name__ == '__main__':
    run()

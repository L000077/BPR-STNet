import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn, optim
from BPR-STNet import net
from Utli import init_weights, Compute_TP_TN_FP_FN, GetData

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

filename = r'data/19_powerDataset.mat'
lablepath = 'E:\LOLpro\lable.xlsx'

band, ydata, subindex = GetData(filename, lablepath)

channel = 31
subject = 19
lr = 1e-3
batch_size = 4
n_epoch = 200
# delta theta alpha beta gamma
tempData = band[4, :, :, :]

result = np.zeros(subject)
matrix = [0, 0, 0, 0]

for i in range(0, subject):
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

    model = net().to(device)
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
            
    #test the result
    model.train(False)
    with torch.no_grad():
        x_test = torch.Tensor(x_test).cuda()
        answer = model(x_test)
        probs = answer.cpu().numpy()
        preds = probs.argmax(axis=-1)
        acc = accuracy_score(y_test, preds)

        print('logSoftMax:', probs)
        print('label:', y_test)
        print('acc:', acc)
        result[i] = acc
        temp_performance = Compute_TP_TN_FP_FN(preds, y_test, matrix)

acc = np.mean(result)
print('current acc:', acc)




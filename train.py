import torch
from model import Model
from dataset import MaskData
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Define the model
    model = Model()
    model.train()

    # hyperparameter
    epoch = 8
    learning_rate = 0.0001
    batchsize = 8

    # Define the loss_function
    loss_func = nn.BCELoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # dataset 불러오기, dataloader 사용
    traindataset = MaskData(root_dir="./dataset", is_train=True)
    # print(traindataset[0])
    dataloader = DataLoader(traindataset, batch_size=batchsize, shuffle=True, drop_last=True)

    loss_arr = []

    for epoch_ in range(epoch):
        for i, (data, label) in enumerate(dataloader):
            optimizer.zero_grad()
            y = model.forward(data)
            loss = loss_func(y, label)
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                loss_arr.append(loss.detach().numpy())

        print(f'{epoch_} : {loss}')

        torch.save(model.state_dict(), './last.pth')

    plt.plot(loss_arr)
    plt.show()




import torch
from model import Model
from dataset import MaskData
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

if __name__ == '__main__':
    correct = 0
    total = 0

    # 학습된 모델 불러오기
    model = Model()
    model.load_state_dict(torch.load('last.pth'))
    model.eval()

    # batch_size
    batchsize = 8

    # dataset 불러오기, dataloader 사용
    testdataset = MaskData(root_dir="./dataset", is_train=False)
    # print(traindataset[0])
    dataloader = DataLoader(testdataset, batch_size=batchsize, shuffle=False, drop_last=True)


    output_value = []

    with torch.no_grad():
        for image, label in dataloader:
            x = image
            y_ = label

            output = model.forward(x)

            for i in range(batchsize):
                if output[i] > 0.5:
                    output_value.append(1)
                else:
                    output_value.append(0)

            output_value = torch.tensor(output_value)

            total += label.size(0)

            for i in range(batchsize):
                if y_[i] == output_value[i]:
                    correct += 1

            output_value = []

            # 테스트 데이터 전체에 대해 위의 작업을 시행한 후 정확도를 구해줍니다.
        print("Accuracy of Test Data: {}%".format(100 * correct / total))


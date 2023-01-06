import glob
import cv2
import torch
from model import Model


correct = 0
total = 0

# 학습된 모델 불러오기
model = Model()
model.load_state_dict(torch.load('last.pth'))
model.eval()

# data 불러오기
data_dirs = glob.glob('testdata*.png')

# label
label = torch.tensor([0, 1, 0]).float()

# output_data

output_datas = []
for i, data_file_name in enumerate(data_dirs):

    img = cv2.imread(data_file_name)

    img = img.transpose(2, 0, 1)

    img = torch.tensor(img).float()

    img /= 255  # normalize

    img = torch.unsqueeze(img,0)

    y = model.forward(img)

    print(f'{i + 1} 번쨰 사진 output 값 : {y}')


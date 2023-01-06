import cv2
import glob
import torch
from torch.utils.data import Dataset
# 사진  tensor
class MaskData(Dataset):
    def __init__(self, root_dir, is_train):
        if is_train:
            type_ = "Train"
        else:
            type_ = "Test"

        image_dirs = glob.glob(f'{root_dir}/{type_}/WithMask/*.png')
        image_dirs += glob.glob(f'{root_dir}/{type_}/WithoutMask/*.png')
        self.image_dir = image_dirs

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, idx):
        image_dir = self.image_dir[idx]
        if 'WithMask' in image_dir:
            label = torch.tensor([1]).float()
        elif 'WithoutMask' in image_dir:
            label = torch.tensor([0]).float()

        img = cv2.imread(image_dir)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128,128))
        img = img.transpose(2,0,1)
        img = torch.tensor(img).float()
        img /= 255   # normalize

        return img, label


if __name__ == '__main__':
    img = cv2.imread('dataset/Test/WithMask/46.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('img.png', img)
    # k = cv2.waitKey(0)
    # cv2.destroyAllWindows()


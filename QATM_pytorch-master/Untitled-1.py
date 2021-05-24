import pandas as pd
from torchvision import models, transforms, utils
import torch
import cv2

thresh_df = pd.read_csv('thresh_template.csv')
print(thresh_df)

transform = None
# 先转成tensor类型，在做normalize
# convert a PIL image to tensor (H*W*C) in range [0,255] to a torch.Tensor(C*H*W) in the range [0.0,1.0]
if not transform:
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        )
    ])
image_raw = cv2.imread("D:\\QATM_pytorch\\QATM_pytorch-master\\sample\\sample1.jpg")
print(image_raw)
image = transform(image_raw).unsqueeze(0)
print(image)

print(image.size())
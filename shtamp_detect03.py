import numpy as np
import torch
import tqdm as tqdm
import albumentations as A
import matplotlib.pyplot as plt
import torchvision.transforms as T
import text_recon01
import easyocr

from PIL import Image
from albumentations.pytorch import ToTensorV2


def shtamp_det(path_imgs):
    model1 = torch.load('/src/ideal_mod9.pth') #map_location=torch.device('cpu'))
    model1.eval()
    IMAGE_HEIGHT = 256*7
    IMAGE_WIDTH = 192*7
    print(path_imgs)
    pred_transform = A.Compose([A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH), ToTensorV2(), ], )
    img = np.array(Image.open(path_imgs).convert("RGB"), dtype=np.float32) / 255
    augmentations = pred_transform(image=img)
    img_result = augmentations["image"]
    img_result = img_result.to('cuda')

    ttt = T.ToPILImage()
    img_T = ttt(img_result)


    print(img_result)
    preds = torch.sigmoid(model1(img_result.unsqueeze(0)))
    preds = (preds > 0.5).float()
    transformation_T = T.ToPILImage()
    imgshow = transformation_T(preds[0])

    bound_box = np.array(imgshow)
    print(bound_box)
    print(bound_box.max())
    print(bound_box.min())
    result1 = np.where(bound_box == 255)
    if (len(result1[0])==0) | (len(result1[1])==0):
        result1 = list(result1)          #преобразуем в список наш кортеж
        result1[0]=[155,125]
        result1[1]=[155,125]
        result1 = tuple(result1)
    print(bound_box[453][265])
    print(max(result1[0]))
    print(max(result1[1]))
    print(min(result1[0]))
    print(min(result1[1]))
    imgshow = Image.composite(img_T,imgshow,imgshow)
    cropped_img = imgshow.crop((min(result1[1]),min(result1[0]),max(result1[1]),max(result1[0])))                       #Вырежем изображение штампика
    cropped_img.save('tryy.png')
    if cropped_img.size[0] < cropped_img.size[1]:
        cropped_img = cropped_img.rotate(-90, expand=True)

    cropp_img1 = cropped_img.crop((cropped_img.size[0]/3.5,0,cropped_img.size[0],cropped_img.size[1]))
    cropp_img1 = cropp_img1.crop((0,cropp_img1.size[0]/8,cropp_img1.size[0],cropp_img1.size[1]))
    cropp_img1 = cropp_img1.crop((0,0,cropp_img1.size[0]/1.05,cropp_img1.size[1]))
    cropp_img1 = cropp_img1.crop((0, 0, cropp_img1.size[0], cropp_img1.size[1]/2))
    cropp_img1.save('doc_num.png')
    reader1 = easyocr.Reader(['ru'],
                             model_storage_directory='/src/easyocr_mod',
                             user_network_directory='/src/easyocr_mod',
                             recog_network='best_accuracy')
    result1 = reader1.readtext('doc_num.png', detail=0)
    return imgshow,cropped_img,cropp_img1,result1

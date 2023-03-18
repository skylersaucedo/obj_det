"""
use this to test obj detection model in pytorch
"""

import torch
import torchvision
from torchvision import models
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def main():

    # load retina model

    #model = models.detection.retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)
    model = models.detection.retinanet_resnet50_fpn(pretrained=True)
    # transforms

    tf_ = T.ToTensor() 

    #load image

    img = Image.open("holdout_image.jpg")

    print(np.shape(img))

    transformed = tf_(img)

    batched = transformed.unsqueeze(0) # model input

    int_img = torch.tensor(transformed * 255, dtype=torch.uint8) # its for our bouding box util

    model = model.eval() # Make sure to not forget this
    with torch.no_grad():
        out = model(batched)

    # make bounding boxes
    from torchvision.utils import draw_bounding_boxes

    score_threshold = .009 #0.1
    first_out = out[0]

    bounding_boxes_img = draw_bounding_boxes(int_img, first_out['boxes'][first_out['scores'] > score_threshold], width=8)

    plt.imshow(bounding_boxes_img.permute(1, 2, 0))
    plt.show()

if __name__ == '__main__':
    main()
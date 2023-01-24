import os
import cv2
import time
import torch
import argparse
import albumentations as A
from pathlib import Path
from models.network import FCDenseNets
from utils.transform import test_transform
from utils.general import localization


@torch.no_grad()
def run(weights, source):
    model_name = Path(weights).stem
    if model_name in FCDenseNets.keys():
        print(f'Loading {model_name}...')
        model = FCDenseNets[model_name].eval().cuda()
    else:
        raise SystemExit('Unsupported type of model')

    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights))
        print('Successfully loaded weights\n')
    else:
        raise SystemExit('Failed to load weights')

    image = cv2.imread(source)
    transformed = test_transform(image=image)
    input_image = torch.unsqueeze(transformed['image'], dim=0).cuda()
    time_stamp = time.time()
    predicted_mask = model(input_image).squeeze()
    print('Inference took %.4fs to complete' % (time.time() - time_stamp))
    predicted_mask = predicted_mask.cpu().numpy() * 255

    _, predicted_mask = cv2.threshold(predicted_mask.astype('uint8'), 0, 255, cv2.THRESH_BINARY)
    resize = A.Resize(image.shape[0], image.shape[1])
    resized = resize(image=image, mask=predicted_mask)
    predicted_mask = resized['mask']
    (x, y), predicted_contours = localization(predicted_mask)
    cv2.drawContours(image, predicted_contours, -1, (0, 0, 255), 3)
    cv2.circle(image, (x, y), 3, (0, 255, 0), 2)

    cv2.imshow('result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', type=str, help='model path', default='weights/FCDenseNet56.pth')
    parser.add_argument('-s', '--source', type=str, help='image source', default='datasets/test/JPEGImages/00001.jpg')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    run(**vars(opt))

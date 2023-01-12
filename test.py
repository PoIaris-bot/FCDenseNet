import os
import cv2
import torch
import argparse
import numpy as np
from math import sqrt
from model import FCDenseNet56, FCDenseNet67, FCDenseNet103
from utils import resize, threshold, transform

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='model name', default='FCDenseNet56')
parser.add_argument('--data_dir', help='directory of keyhole dataset', default='data/test')
parser.add_argument('--weight_dir', help='directory of FCDenseNet parameters', default='param')
args = parser.parse_args()

if __name__ == '__main__':
    fc_dense_net = None
    if args.model == 'FCDenseNet56':
        fc_dense_net = FCDenseNet56().eval().cuda()
    elif args.model == 'FCDenseNet67':
        fc_dense_net = FCDenseNet67().eval().cuda()
    else:
        fc_dense_net = FCDenseNet103().eval().cuda()

    weight_path = os.path.join(args.weight_dir, f'{args.model}.pth')
    if os.path.exists(weight_path):
        fc_dense_net.load_state_dict(torch.load(weight_path))
        print('Successfully loaded weights')
    else:
        print('Failed to load weights')

    image_dir = os.path.join(args.data_dir, 'JPEGImages')
    segment_dir = os.path.join(args.data_dir, 'SegmentationClass')
    image_names = os.listdir(image_dir)

    avg_error = 0
    max_error = 0
    for image_name in image_names:
        image_path = os.path.join(image_dir, image_name)
        segment_image_path = os.path.join(segment_dir, image_name)
        image = resize(cv2.imread(image_path))
        segment_image = resize(cv2.imread(segment_image_path, 0))

        input_image = torch.unsqueeze(transform(image), dim=0).cuda()
        output_image = fc_dense_net(input_image)

        output_image = output_image.cpu().detach().numpy().reshape(output_image.shape[-2:]) * 255
        output_binary = threshold(output_image.astype('uint8'))
        binary = threshold(segment_image)
        output_contours, _ = cv2.findContours(output_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, output_contours, -1, (0, 255, 0), 3)
        cv2.drawContours(image, contours, -1, (0, 0, 255), 3)

        y, x = map(int, np.mean(np.where(output_binary > 0), axis=1))
        y0, x0 = map(int, np.mean(np.where(binary > 0), axis=1))
        cv2.circle(image, (x, y), 3, (0, 255, 0), 2)
        cv2.circle(image, (x0, y0), 3, (0, 0, 255), 2)
        if not os.path.exists('result'):
            os.makedirs('result')
        cv2.imwrite(f'result/{image_name}', image)

        error = sqrt((x - x0) ** 2 + (y - y0) ** 2)
        avg_error += error
        max_error = error if error > max_error else max_error
    print(f'average error: {avg_error / len(image_names)} maximum error: {max_error}')

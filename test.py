import os
import cv2
import time
import torch
import argparse
import numpy as np
from math import sqrt
from pathlib import Path
from models.network import FCDenseNets
from utils.transform import resize, transform


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

    image_dir = os.path.join(source, 'JPEGImages')
    segment_dir = os.path.join(source, 'SegmentationClass')
    image_names = os.listdir(image_dir)

    save_dir = 'runs/test/exp'
    if os.path.exists(save_dir):
        for n in range(2, 9999):
            temp_dir = f'{save_dir}{n}'
            if not os.path.exists(temp_dir):
                save_dir = temp_dir
                break
    os.makedirs(save_dir)

    avg_error = 0
    max_error = 0
    max_error_image_name = ''

    error_leq1p_count = 0
    error_leq3p_count = 0
    error_leq5p_count = 0

    avg_infer_time = 0
    for image_name in image_names:
        image_path = os.path.join(image_dir, image_name)
        segment_image_path = os.path.join(segment_dir, image_name)
        image = resize(cv2.imread(image_path))
        segment_image = resize(cv2.imread(segment_image_path, 0))

        input_image = torch.unsqueeze(transform(image), dim=0).cuda()
        start = time.time()
        output_image = model(input_image)
        end = time.time()
        avg_infer_time += end - start
        output_image = output_image.cpu().detach().numpy().reshape(output_image.shape[-2:]) * 255

        _, output_binary = cv2.threshold(output_image.astype('uint8'), 5, 255, cv2.THRESH_BINARY)
        _, binary = cv2.threshold(segment_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        output_contours, _ = cv2.findContours(output_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        output_area = []
        for i in range(len(output_contours)):
            output_area.append(cv2.contourArea(output_contours[i]))
        output_max_idx = np.argmax(output_area)
        for i in range(len(output_contours)):
            if i != output_max_idx:
                cv2.fillPoly(output_binary, [output_contours[i]], 0)

        cv2.drawContours(image, output_contours, output_max_idx, (0, 255, 0), 3)
        cv2.drawContours(image, contours, -1, (0, 0, 255), 3)

        y, x = map(int, np.mean(np.where(output_binary > 0), axis=1))
        y0, x0 = map(int, np.mean(np.where(binary > 0), axis=1))
        cv2.circle(image, (x, y), 3, (0, 255, 0), 2)
        cv2.circle(image, (x0, y0), 3, (0, 0, 255), 2)
        cv2.imwrite(f'{save_dir}/{image_name}', image)

        error = sqrt((x - x0) ** 2 + (y - y0) ** 2)

        error_leq1p_count += 1 if error <= 1 else 0
        error_leq3p_count += 1 if error <= 3 else 0
        error_leq5p_count += 1 if error <= 5 else 0

        avg_error += error
        max_error_image_name = image_name if error > max_error else max_error_image_name
        max_error = error if error > max_error else max_error
    print(f'average error: {avg_error / len(image_names)} maximum error: {max_error}')
    print(f'average inference time: {avg_infer_time / len(image_names)} s')
    print(f'percentage of images with error less equal than 1 pixel: {error_leq1p_count / len(image_names)}')
    print(f'percentage of images with error less equal than 3 pixels: {error_leq3p_count / len(image_names)}')
    print(f'percentage of images with error less equal than 5 pixels: {error_leq5p_count / len(image_names)}')
    print(f'test image with the maximum error: {max_error_image_name}')
    print(f'\nResults saved to {save_dir}')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', type=str, help='model path', default='weights/FCDenseNet56.pth')
    parser.add_argument('-s', '--source', type=str, help='image source', default='datasets/test')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    run(**vars(opt))

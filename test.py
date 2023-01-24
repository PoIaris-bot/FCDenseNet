import os
import cv2
import time
import torch
import argparse
import albumentations as A
from math import sqrt
from pathlib import Path
from models.network import FCDenseNets
from utils.general import increment_path, localization
from utils.transform import test_transform


def create_model(weights, device):
    model_name = Path(weights).stem
    if model_name in FCDenseNets.keys():
        print(f'Loading {model_name}...')
        model = FCDenseNets[model_name].eval().to(device)
    else:
        raise SystemExit('Unsupported type of model')

    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights))
        print('Successfully loaded weights\n')
    else:
        raise SystemExit('Failed to load weights')
    return model


def test(model, test_dataset, device, save_dir):
    image_directory = os.path.join(test_dataset, 'JPEGImages')
    mask_directory = os.path.join(test_dataset, 'SegmentationClass')
    image_filenames = os.listdir(image_directory)

    avg_error = 0
    max_error = 0
    max_error_image_name = ''

    error_leq1p_count = 0
    error_leq3p_count = 0
    error_leq5p_count = 0

    avg_infer_time = 0
    for image_filename in image_filenames:
        image_path = os.path.join(image_directory, image_filename)
        mask_path = os.path.join(mask_directory, image_filename)

        image = cv2.imread(image_path)
        original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        transformed = test_transform(image=image)
        input_image = torch.unsqueeze(transformed['image'], dim=0).to(device)
        start = time.time()
        predicted_mask = model(input_image).squeeze()
        end = time.time()
        avg_infer_time += end - start
        predicted_mask = predicted_mask.cpu().numpy() * 255

        _, predicted_mask = cv2.threshold(predicted_mask.astype('uint8'), 5, 255, cv2.THRESH_BINARY)
        _, original_mask = cv2.threshold(original_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        resize = A.Resize(image.shape[0], image.shape[1])
        resized = resize(image=image, mask=predicted_mask)
        predicted_mask = resized['mask']

        (x, y), predicted_contours = localization(predicted_mask)
        (x0, y0), original_contours = localization(original_mask)

        cv2.drawContours(image, predicted_contours, -1, (0, 0, 255), 3)
        cv2.drawContours(image, original_contours, -1, (0, 255, 0), 3)

        cv2.circle(image, (x, y), 3, (0, 255, 0), 2)
        cv2.circle(image, (x0, y0), 3, (0, 0, 255), 2)
        cv2.imwrite(f'{save_dir}/{image_filename}', image)

        error = sqrt((x - x0) ** 2 + (y - y0) ** 2)

        error_leq1p_count += 1 if error <= 1 else 0
        error_leq3p_count += 1 if error <= 3 else 0
        error_leq5p_count += 1 if error <= 5 else 0

        avg_error += error
        max_error_image_name = image_filename if error > max_error else max_error_image_name
        max_error = error if error > max_error else max_error
    print(f'average error: {avg_error / len(image_filenames)} maximum error: {max_error}')
    print(f'average inference time: {avg_infer_time / len(image_filenames)} s')
    print(f'percentage of images with error less equal than 1 pixel: {error_leq1p_count / len(image_filenames)}')
    print(f'percentage of images with error less equal than 3 pixels: {error_leq3p_count / len(image_filenames)}')
    print(f'percentage of images with error less equal than 5 pixels: {error_leq5p_count / len(image_filenames)}')
    print(f'test image with the maximum error: {max_error_image_name}')
    print(f'\nResults saved to {save_dir}')


@torch.no_grad()
def run(weights, test_dataset, device):
    save_dir = increment_path('runs/test/exp')
    os.makedirs(save_dir)

    model = create_model(weights, device)
    test(model, test_dataset, device, save_dir)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', type=str, help='model path', default='weights/FCDenseNet56.pth')
    parser.add_argument('-t', '--test-dataset', type=str, help='test datasets', default='datasets/test')
    parser.add_argument('-d', '--device', type=str, help='device', default='cuda')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    run(**vars(opt))

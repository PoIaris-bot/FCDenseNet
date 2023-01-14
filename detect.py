import os
import cv2
import time
import torch
import argparse
import numpy as np
from pathlib import Path
from openvino.inference_engine import IECore
from model import FCDenseNets
from utils import resize, transform


@torch.no_grad()
def run(weights, source):
    model_format = Path(weights).suffix
    if model_format == '.pth':
        model_name = Path(weights).stem
        if model_name in FCDenseNets.keys():
            model = FCDenseNets[model_name].eval().cuda()
        else:
            raise SystemExit('Unsupported type of model')

        if os.path.exists(weights):
            model.load_state_dict(torch.load(weights))
            print('Successfully loaded weights')
        else:
            raise SystemExit('Failed to load weights')
    elif model_format == '.onnx':
        ie = IECore()
        net = ie.read_network(model=weights)
        input_blob = next(iter(net.input_info))
        output_blob = next(iter(net.outputs))
        model = ie.load_network(network=net, device_name='CPU')
    else:
        raise SystemExit('Unsupported type of model format')

    image = resize(cv2.imread(source))
    if model_format == '.pth':
        input_image = torch.unsqueeze(transform(image), dim=0).cuda()
        time_stamp = time.time()
        output_image = model(input_image)
        print('Inference took %.4fs to complete' % (time.time() - time_stamp))
        output_image = output_image.cpu().detach().numpy()
    else:
        input_image = image.transpose(2, 0, 1) / 255
        time_stamp = time.time()
        output = model.infer(inputs={input_blob: [input_image]})
        print('Inference took %.4fs to complete' % (time.time() - time_stamp))
        output_image = output[output_blob]

    output_image = output_image.reshape(output_image.shape[-2:]) * 255
    _, binary = cv2.threshold(output_image.astype('uint8'), 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    max_idx = np.argmax(area)
    for i in range(len(contours)):
        if i != max_idx:
            cv2.fillPoly(binary, [contours[i]], 0)
    cv2.drawContours(image, contours, max_idx, (0, 255, 0), 3)

    y, x = map(int, np.mean(np.where(binary > 0), axis=1))
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

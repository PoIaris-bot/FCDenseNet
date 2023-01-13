import os
import cv2
import time
import torch
import argparse
import numpy as np
from openvino.inference_engine import IECore
from model import FCDenseNet56, FCDenseNet67, FCDenseNet103
from utils import resize, transform

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='model name', default='FCDenseNet56')
parser.add_argument('-f', '--format', help='format of FCDenseNet', default='torch')
parser.add_argument('--model_dir', help='directory of onnx model', default='model')
parser.add_argument('-i', '--image_path', help='path of keyhole image', default='data/test/JPEGImages/00001.jpg')
parser.add_argument('--weight_dir', help='directory of FCDenseNet parameters', default='param')
args = parser.parse_args()

if __name__ == '__main__':
    if args.format == 'torch':
        if args.model == 'FCDenseNet56':
            fc_dense_net = FCDenseNet56()
        elif args.model == 'FCDenseNet67':
            fc_dense_net = FCDenseNet67()
        elif args.model == 'FCDenseNet103':
            fc_dense_net = FCDenseNet103()
        else:
            raise SystemExit('Wrong type of network model')

        weight_path = os.path.join(args.weight_dir, f'{args.model}.pth')
        if os.path.exists(weight_path):
            fc_dense_net.load_state_dict(torch.load(weight_path))
            print('Successfully loaded weights')
        else:
            raise SystemExit('Failed to load weights')
        fc_dense_net = fc_dense_net.eval().cuda()
    elif args.format == 'onnx':
        ie = IECore()
        net = ie.read_network(model=os.path.join(args.model_dir, f'{args.model}.onnx'))
        input_blob = next(iter(net.input_info))
        output_blob = next(iter(net.outputs))
        fc_dense_net = ie.load_network(network=net, device_name='CPU')
    else:
        raise SystemExit('Wrong type of network format')

    image = resize(cv2.imread(args.image_path))
    if args.format == 'torch':
        input_image = torch.unsqueeze(transform(image), dim=0).cuda()
        time_stamp = time.time()
        output_image = fc_dense_net(input_image)
        print('Inference took %.4fs to complete' % (time.time() - time_stamp))
        output_image = output_image.cpu().detach().numpy().reshape(output_image.shape[-2:]) * 255
    else:
        input_image = image.transpose(2, 0, 1) / 255
        time_stamp = time.time()
        output = fc_dense_net.infer(inputs={input_blob: [input_image]})
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

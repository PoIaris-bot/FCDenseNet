import os
import torch
import argparse
from pathlib import Path
from model import FCDenseNets


@torch.no_grad()
def run(weights):
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

    dummy_input = torch.randn(1, 3, 256, 256).cuda()
    torch.onnx.export(model, dummy_input, os.path.join(os.path.split(weights)[0], f'{model_name}.onnx'), verbose=True)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', type=str, help='model path', default='weights/FCDenseNet56.pth')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    run(**vars(opt))

import os
import torch
import argparse
from model import FCDenseNet56, FCDenseNet67, FCDenseNet103

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='model name', default='FCDenseNet56')
parser.add_argument('--model_dir', help='directory of onnx model', default='model')
parser.add_argument('--weight_dir', help='directory of FCDenseNet parameters', default='param')
args = parser.parse_args()

if __name__ == '__main__':
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
        print('Failed to load weights')

    fc_dense_net = fc_dense_net.eval().cuda()
    dummy_input = torch.randn(1, 3, 256, 256).cuda()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    torch.onnx.export(fc_dense_net, dummy_input, os.path.join(args.model_dir, f'{args.model}.onnx'), verbose=True)

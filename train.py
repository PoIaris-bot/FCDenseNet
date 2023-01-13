import os
import torch
import argparse
from torch import nn, optim
from torch.utils.data import DataLoader
from model import FCDenseNet56, FCDenseNet67, FCDenseNet103
from dataset import KeyholeDataset

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='model name', default='FCDenseNet56')
parser.add_argument('--data_dir', help='directory of keyhole dataset', default='data/train')
parser.add_argument('--weight_dir', help='directory of FCDenseNet parameters', default='param')
parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=4)
parser.add_argument('-e', '--epochs', help='number of epochs', type=int, default=10)
args = parser.parse_args()

if __name__ == '__main__':
    data_loader = DataLoader(KeyholeDataset(args.data_dir), batch_size=args.batch_size, num_workers=4, shuffle=True)
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
        if not os.path.exists(args.weight_dir):
            os.makedirs(args.weight_dir)
        print('Failed to load weights')
    fc_dense_net = fc_dense_net.cuda()
    
    opt = optim.Adam(fc_dense_net.parameters())
    loss_func = nn.BCELoss()

    epoch = 1
    while epoch <= args.epochs:
        loss = 0
        avg_loss = 0
        for batch, (image, segment_image) in enumerate(data_loader):
            image, segment_image = image.cuda(), segment_image.cuda()

            output_image = fc_dense_net(image)
            train_loss = loss_func(output_image, segment_image)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            loss = train_loss.item()
            avg_loss += loss
            if batch % 100 == 0:
                print('\repoch: {:>5d}/{:<5d} batch: {:>5d}/{:<5d} loss: {:^12.8f} average loss: {:^12.8f}'.format(
                    epoch, args.epochs, batch, len(data_loader) - 1, loss, avg_loss / (batch + 1)
                ), end='')
                torch.save(fc_dense_net.state_dict(), weight_path)

        torch.save(fc_dense_net.state_dict(), weight_path)
        print('\repoch: {:>5d}/{:<5d} batch: {:>5d}/{:<5d} loss: {:^12.8f} average loss: {:^12.8f}'.format(
            epoch, args.epochs, len(data_loader) - 1, len(data_loader) - 1, loss, avg_loss / len(data_loader)
        ), end='\n')
        epoch += 1

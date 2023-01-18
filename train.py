import os
import torch
import argparse
from torch import nn, optim
from torch.utils.data import DataLoader
from models.network import FCDenseNets
from utils.dataset import KeyholeDataset


def run(model_name, weights, source, augment, batch_size, num_workers, epochs):
    data_loader = DataLoader(
        KeyholeDataset(source, augment), batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    if model_name in FCDenseNets.keys():
        model = FCDenseNets[model_name].cuda()
    else:
        raise SystemExit('Unsupported type of model')

    if weights and os.path.exists(weights):
        model.load_state_dict(torch.load(weights))
        print('Successfully loaded weights')

    save_dir = 'runs/train/exp'
    if os.path.exists(save_dir):
        for n in range(2, 9999):
            temp_dir = f'{save_dir}{n}'
            if not os.path.exists(temp_dir):
                save_dir = temp_dir
                break
    os.makedirs(save_dir)

    optimizer = optim.Adam(model.parameters())
    loss_func = nn.BCELoss()

    epoch = 0
    while epoch < epochs:
        loss = 0
        avg_loss = 0
        for batch, (image, segment_image) in enumerate(data_loader):
            image, segment_image = image.cuda(), segment_image.cuda()

            output_image = model(image)
            train_loss = loss_func(output_image, segment_image)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            loss = train_loss.item()
            avg_loss += loss
            if batch % 100 == 0:
                print('\repoch: {:>5d}/{:<5d} batch: {:>5d}/{:<5d} loss: {:^12.8f} average loss: {:^12.8f}'.format(
                    epoch, epochs - 1, batch, len(data_loader) - 1, loss, avg_loss / (batch + 1)
                ), end='')
                torch.save(model.state_dict(), f'{save_dir}/{model_name}.pth')

        torch.save(model.state_dict(), f'{save_dir}/{model_name}.pth')
        print('\repoch: {:>5d}/{:<5d} batch: {:>5d}/{:<5d} loss: {:^12.8f} average loss: {:^12.8f}'.format(
            epoch, epochs - 1, len(data_loader) - 1, len(data_loader) - 1, loss, avg_loss / len(data_loader)
        ), end='\n')
        epoch += 1
    print(f'\nResults saved to {save_dir}')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-name', help='model name', default='FCDenseNet56')
    parser.add_argument('-s', '--source', type=str, help='image source', default='datasets/train')
    parser.add_argument('-w', '--weights', help='weight path', default='')
    parser.add_argument('-b', '--batch-size', type=int, help='batch size', default=4)
    parser.add_argument('-n', '--num-workers', type=int, help='number of workers', default=4)
    parser.add_argument('-a', '--augment', type=bool, help='image augmentation', default=False)
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs', default=50)
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    run(**vars(opt))

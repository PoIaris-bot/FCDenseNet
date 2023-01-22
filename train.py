import os
import torch
import argparse
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from models.network import FCDenseNets
from utils.dataset import KeyholeDataset


def run(model_name, weights, train_data, val_data, augment, batch_size, num_workers, epochs):
    train_data_loader = DataLoader(
        KeyholeDataset(train_data, augment), batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    train_data_count = len(train_data_loader)
    val_data_loader = DataLoader(
        KeyholeDataset(val_data, False), batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    val_data_count = len(val_data_loader)

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
    min_avg_val_loss = np.inf
    while epoch < epochs:
        # training
        loss = 0
        avg_train_loss = 0
        model.train()
        for batch, (image, segment_image) in enumerate(train_data_loader):
            image, segment_image = image.cuda(), segment_image.cuda()

            output_image = model(image)
            train_loss = loss_func(output_image, segment_image)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            loss = train_loss.item()
            avg_train_loss += loss
            if batch % 100 == 0:
                print('\r[train] epoch: {:>5d}/{:<5d} batch: {:>5d}/{:<5d} loss: {:^12.8f} avg_loss: {:^12.8f}'.format(
                    epoch, epochs - 1, batch, train_data_count - 1, loss, avg_train_loss / (batch + 1)
                ), end='')

        torch.save(model.state_dict(), f'{save_dir}/last.pth')
        print('\r[train] epoch: {:>5d}/{:<5d} batch: {:>5d}/{:<5d} loss: {:^12.8f} avg_loss: {:^12.8f}'.format(
            epoch, epochs - 1, train_data_count - 1, train_data_count - 1, loss, avg_train_loss / train_data_count
        ), end='\n')

        # validation
        avg_val_loss = 0
        model.eval()
        with torch.no_grad():
            for image, segment_image in val_data_loader:
                image, segment_image = image.cuda(), segment_image.cuda()

                output_image = model(image)
                val_loss = loss_func(output_image, segment_image)

                avg_val_loss += val_loss.item()
        avg_val_loss /= val_data_count
        if avg_val_loss < min_avg_val_loss:
            min_avg_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'{save_dir}/best.pth')
        print('[validation] epoch: {:>5d}/{:<5d} avg_loss: {:^12.8f} min_avg_loss: {:^12.8f}'.format(
            epoch, epochs - 1, avg_val_loss, min_avg_val_loss
        ), end='\n')

        epoch += 1
    print(f'\nResults saved to {save_dir}')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-name', help='model name', default='FCDenseNet56')
    parser.add_argument('-t', '--train-data', type=str, help='training datasets', default='datasets/train')
    parser.add_argument('-v', '--val-data', type=str, help='validation datasets', default='datasets/val')
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

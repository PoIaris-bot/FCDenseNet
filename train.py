import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from models.network import FCDenseNets
from utils.dataset import KeyholeDataset
from utils.general import MetricMonitor, increment_path
from utils.transform import train_transform, val_transform


def train(train_loader, model, criterion, optimizer, epoch, epochs, device):
    metric_monitor = MetricMonitor(float_precision=5)
    model.train()
    stream = tqdm(train_loader)
    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(images).squeeze(1)
        loss = criterion(output, target)
        metric_monitor.update("Loss", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stream.set_description(
            "Epoch: {epoch}/{epochs}. Train.      {metric_monitor}".format(
                epoch=epoch, epochs=epochs, metric_monitor=metric_monitor
            )
        )


def validate(val_loader, model, criterion, epoch, epochs, device):
    metric_monitor = MetricMonitor(float_precision=5)
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(images).squeeze(1)
            loss = criterion(output, target)
            metric_monitor.update("Loss", loss.item())
            stream.set_description(
                "Epoch: {epoch}/{epochs}. Validation. {metric_monitor}".format(
                    epoch=epoch, epochs=epochs, metric_monitor=metric_monitor
                )
            )
    avg_loss = metric_monitor.metrics['Loss']['avg']
    return avg_loss


def create_model(model_name, weights, device):
    assert model_name in FCDenseNets.keys()
    model = FCDenseNets[model_name]

    if weights and os.path.exists(weights):
        model.load_state_dict(torch.load(weights))
    return model.to(device)


def train_and_validate(model, train_dataset, val_dataset, device, batch_size, num_workers, epochs, save_dir):
    train_loader = DataLoader(
        KeyholeDataset(train_dataset, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        KeyholeDataset(val_dataset, val_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    min_loss = np.inf
    for epoch in range(1, epochs + 1):
        train(train_loader, model, criterion, optimizer, epoch, epochs, device)
        loss = validate(val_loader, model, criterion, epoch, epochs, device)

        torch.save(model.state_dict(), f'{save_dir}/last.pth')
        if loss < min_loss:
            min_loss = loss
            torch.save(model.state_dict(), f'{save_dir}/best.pth')
    print(f'\nResults saved to {save_dir}')


def run(model_name, weights, train_dataset, val_dataset, device, batch_size, num_workers, epochs):
    save_dir = increment_path('runs/train/exp')
    os.makedirs(save_dir, exist_ok=True)

    model = create_model(model_name, weights, device)
    train_and_validate(model, train_dataset, val_dataset, device, batch_size, num_workers, epochs, save_dir)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-name', help='model name', default='FCDenseNet56')
    parser.add_argument('-t', '--train-dataset', type=str, help='training datasets', default='datasets/train')
    parser.add_argument('-v', '--val-dataset', type=str, help='validation datasets', default='datasets/val')
    parser.add_argument('-w', '--weights', help='weight path', default='')
    parser.add_argument('-b', '--batch-size', type=int, help='batch size', default=4)
    parser.add_argument('-n', '--num-workers', type=int, help='number of workers', default=4)
    parser.add_argument('-d', '--device', type=str, help='device', default='cuda')
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs', default=50)
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    run(**vars(opt))

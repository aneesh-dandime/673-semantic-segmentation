# imports
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from Code.unet import UNet

from Code.data import KittiDataset
from training import train

def count_params(model):
    params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params, train_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='The name of the experiment.')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='The number of epochs. Default: 100')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5,
                        help='The learning rate. Default: 1e-5')
    parser.add_argument('-etc', '--epochs_till_chkpt', type=int, default=1,
                        help='The number of epochs after which model is saved. Default: 1')
    parser.add_argument('-bs', '--batch_size', type=int, default=100,
                        help='The batch size. Default: 100')
    parser.add_argument('-md', '--model_dir', type=str, default='../Data/kitti',
                        help='The folder where the dataset is stored. Default: ../Data/kitti')
    args = parser.parse_args()

    # Params
    name = args.name
    epochs = args.epochs
    lr = args.learning_rate
    epochs_till_chkpt = args.epochs_till_chkpt
    batch_size = args.batch_size
    model_dir = args.model_dir

    # Loss function
    loss_func = nn.CrossEntropyLoss()

    # Model
    model = UNet(n_channels=3, n_classes=3)
    params, train_params = count_params(model)
    print('===========================================================')
    print(f'Starting training...')
    print(f'Number of model parameters: {params}')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print('CUDA enabled GPU found!')
        model = model.to(device)
    else:
        print('CUDA enabled GPU not found! Using CPU.')
    print('===========================================================')

    # Dataset and dataloader
    dataset = KittiDataset(os.path.join(model_dir, 'training'), shape=(621, 187))
    total_length = len(dataset)
    train_length = int(0.8 * total_length)
    val_length = total_length - train_length
    train_dataset, val_dataset = random_split(dataset,[train_length, val_length])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Train!
    train(model, train_dataloader, epochs, lr, epochs_till_chkpt, model_dir, name,
        loss_func, validation_dataloader=val_dataloader)

from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
import torch
import os
import sys
import torchvision
from torchvision.transforms import ToTensor
from helper import LoadImageFromFolder, AverageMeter
from torch.utils.data.dataloader import DataLoader
from Model_PartialConv.partialconvNN import ImageinpaintingPConvNN
from Model_CNN import InpaintingCNN

# For utilities
import time
import numpy as np
import torch.nn as nn

best_losses_partialConv = float("inf")
best_losses_vanillaCNN = float("inf")
use_gpu = torch.cuda.is_available()


def train(train_loader, model, criterion, optimizer, epoch, model_type):
    print('Starting  training epoch {}'.format(epoch))
    model.train()

    batch_time, losses = AverageMeter(), AverageMeter()

    for i, (X_masked, Mask, Y_unmasked) in enumerate(train_loader):
        X_masked = X_masked.permute(0, 3, 2, 1)
        Mask = Mask.permute(0, 3, 2, 1)
        Y_unmasked = Y_unmasked.permute(0, 3, 2, 1)

        if use_gpu:
            X_masked, Mask, Y_unmasked = X_masked.cuda(), Mask.cuda(), Y_unmasked.cuda()

        X_masked = X_masked.float()
        Mask = Mask.float()
        Y_unmasked = Y_unmasked.float()

        if model_type == 'PartialConv':
            output = model((X_masked, Mask))
        else:
            output = model(X_masked)

        loss = criterion(output, Y_unmasked)
        losses.update(loss.item(), X_masked.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print model accuracy -- in the code below, val refers to value, not validation
        if i % 20 == 0:
            print(model_type + ' Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time, loss=losses))

    print('Finished training epoch {}'.format(epoch))


def validation(valid_loader, model, criterion, epoch, model_type):
    model.eval()
    # Prepare value counters and timers
    batch_time, losses = AverageMeter(), AverageMeter()

    end = time.time()

    for i, (X_masked, Mask, Y_unmasked) in enumerate(valid_loader):
        X_masked = X_masked.permute(0, 3, 2, 1)
        Mask = Mask.permute(0, 3, 2, 1)
        Y_unmasked = Y_unmasked.permute(0, 3, 2, 1)

        if use_gpu:
            X_masked, Mask, Y_unmasked = X_masked.cuda(), Mask.cuda(), Y_unmasked.cuda()

        X_masked = X_masked.float()
        Mask = Mask.float()
        Y_unmasked = Y_unmasked.float()

        if model_type == 'PartialConv':
            output = model((X_masked, Mask))
        else:
            output = model(X_masked)

        loss = criterion(output, Y_unmasked)
        losses.update(loss.item(), X_masked.size(0))

        # Record time to do forward passes and save images
        batch_time.update(time.time() - end)
        end = time.time()

        # Print model accuracy -- in the code below, val refers to both value and validation
        if i % 10 == 0:
            print(model_type + 'Validate: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch,
                                                                  i, len(valid_loader), batch_time=batch_time,
                                                                  loss=losses))

    print('Finished validation.')
    return losses.avg


def test(test_loader, model, criterion, model_type):
    model.eval()
    # Prepare value counters and timers
    batch_time, losses = AverageMeter(), AverageMeter()

    end = time.time()
    c = 0
    for i, (X_masked, Mask, Y_unmasked) in enumerate(test_loader):
        X_masked = X_masked.permute(0, 3, 2, 1)
        Mask = Mask.permute(0, 3, 2, 1)
        Y_unmasked = Y_unmasked.permute(0, 3, 2, 1)

        if use_gpu:
            X_masked, Mask, Y_unmasked = X_masked.cuda(), Mask.cuda(), Y_unmasked.cuda()

        X_masked = X_masked.float()
        Mask = Mask.float()
        Y_unmasked = Y_unmasked.float()

        if model_type == 'PartialConv':
            output = model((X_masked, Mask))
        else:
            output = model(X_masked)

        loss = criterion(output, Y_unmasked)
        losses.update(loss.item(), X_masked.size(0))

        # Save images to file

        for j in range(10):
            img1 = output[j]
            img2 = X_masked[j]

            save_name = 'img-{}.jpg'.format(c)
            if model_type == 'PartialConv':
                save_image(img1, "PredictedOutput_PartialConv/" + save_name)
            else:
                save_image(img1, "PredictedOutput_VanillaCNN/" + save_name)

            save_image(img2, "InputImage/" + save_name)
            c += 1
        # Record time to do forward passes and save images
        batch_time.update(time.time() - end)
        end = time.time()

        # Print model accuracy -- in the code below, val refers to both value and validation
        if i % 10 == 9:
            print(model_type + ' Testing: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                i, len(test_loader), batch_time=batch_time, loss=losses))

    print('Finished testing.')


def main(load_pretrained):
    os.makedirs("InputImage", exist_ok=True)
    os.makedirs("PredictedOutput_PartialConv", exist_ok=True)
    os.makedirs("PredictedOutput_VanillaCNN", exist_ok=True)

    global best_losses_partialConv, best_losses_vanillaCNN

    train_data = CIFAR10(root='data/', download=True, transform=ToTensor())
    test_data = CIFAR10(root='data/', train=False, download=True, transform=ToTensor())

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = torch.utils.data.RandomSampler(train_idx)
    valid_sampler = torch.utils.data.RandomSampler(valid_idx)

    dataset_train = LoadImageFromFolder(train_data.data)
    dataset_test = LoadImageFromFolder(test_data.data)

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=128, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=128, sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, shuffle=False, batch_size=128)

    modelPartialConv = ImageinpaintingPConvNN()
    modelVanillaCNN = InpaintingCNN()

    criterion = nn.MSELoss()
    if use_gpu:
        modelPartialConv = modelPartialConv.cuda()
        modelVanillaCNN = modelVanillaCNN.cuda()
        criterion = criterion.cuda()
        print('Loaded model onto GPU.')

    optimizer = torch.optim.Adam(modelPartialConv.parameters(), lr=1e-2, weight_decay=0.0)
    epochs = 100
    best_modelPartialConv = modelPartialConv
    best_modelVanillaCNN = modelVanillaCNN

    if not load_pretrained:
        os.makedirs("Checkpoints", exist_ok=True)
        for epoch in range(epochs):
            # Train for one epoch
            train(train_loader, modelPartialConv, criterion, optimizer, epoch, 'PartialConv')
            train(train_loader, modelVanillaCNN, criterion, optimizer, epoch, 'VanillaCNN')

            with torch.no_grad():
                lossesPartialConv = validation(valid_loader, modelPartialConv, criterion, epoch, 'PartialConv')
                lossesVanillaCNN = validation(valid_loader, modelVanillaCNN, criterion, epoch, 'VanillaCNN')

            if lossesPartialConv < best_losses_partialConv:
                best_losses_partialConv = lossesPartialConv
                best_modelPartialConv = modelPartialConv
                torch.save(best_modelPartialConv.state_dict(),
                           'Checkpoints/' + 'PartialCovModel_Of_Epoch-{}'.format(epoch))

            if lossesVanillaCNN < best_losses_vanillaCNN:
                best_losses_vanillaCNN = lossesVanillaCNN
                best_modelVanillaCNN = modelVanillaCNN
                torch.save(best_modelVanillaCNN.state_dict(),
                           'Checkpoints/' + 'VanillaCNNModel_Of_Epoch-{}'.format(epoch))

        print('Completed Execution - Partial Convolution. Best loss: ', best_losses_partialConv)
        print('Completed Execution - VanillaCNN. Best loss: ', best_losses_vanillaCNN)

    else:
        best_modelPartialConv.load_state_dict(
            torch.load('Checkpoints/model-partialConv', map_location=torch.device('cpu')))
        best_modelVanillaCNN.load_state_dict(
            torch.load('Checkpoints/model-VanillaCNN', map_location=torch.device('cpu')))

    test(test_loader, best_modelPartialConv, criterion, 'PartialConv')
    test(test_loader, best_modelVanillaCNN, criterion, 'VanillaCNN')


if __name__ == '__main__':
    val = sys.argv[1]
    load_pretrained = True if int(val) == 1 else False
    print('Load PreTrained Model : ', load_pretrained)
    main(load_pretrained)

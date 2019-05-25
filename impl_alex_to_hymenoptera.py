#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torchvision.models.alexnet as alexnet
import torchvision
import os

from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch
import torch.nn as nn
import imagenet.main as imagenet_util
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import torchvision.datasets as datasets


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 返回训练集，测试集，以及数据标签类
def load_hymenoptera():
    data_dir = 'data/hymenoptera_data'
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data_transforms = {
        'train': train_transform,
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    return dataloaders['train'],dataloaders['val'],class_names

def train_and_valide_on_hymenoptera(args):
    train_loader,val_loader,classes = load_hymenoptera()
    model = alexnet(pretrained = True)
    model.classifier[6]=nn.Linear(4096,2,bias=True)
    model = model.cuda(args.gpu)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    best_acc1 = -1
    ngpus_per_node = torch.cuda.device_count()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        imagenet_util.train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = imagenet_util.validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            imagenet_util.save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)

# 展示图片以及预测出的label
def imshow(img,title):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

def test_best_model_on_hymenoptera(args):
    train_loader,val_loader,classes = load_hymenoptera()
    model = alexnet(pretrained = False)
    model.classifier[6]=nn.Linear(4096,2,bias=True)
    model = model.cuda(args.gpu)
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args.resume, checkpoint['epoch']))
    dataiter = iter(val_loader)
    images,labels = dataiter.next()
    classes_names = ' '.join('%5s' % classes[labels[j]] for j in range(4))

    images = images.to('cuda:0')
    output = model(images)
    scores, preds= torch.max(output, 1)
    # preds = preds.squeeze(0)
    preds = preds.to('cpu')
    preds_names = ' '.join('%5s' % classes[preds[j]] for j in range(4))
    images = images.to('cpu')
    title = str(classes_names)
    title += str(preds_names)
    imshow(torchvision.utils.make_grid(images),title)

if __name__ == '__main__':
    args = imagenet_util.parser.parse_args(sys.argv)
    print(args)
    # train_and_valide_on_hymenoptera(args)
    test_best_model_on_hymenoptera(args)

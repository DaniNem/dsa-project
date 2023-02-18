from __future__ import print_function

import os.path
import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn

from torchvision import transforms, datasets
from util import AverageMeter
from networks.resnet_big import SupConResNet

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')



    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'celeba', 'cifar100'], help='dataset')



    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.features_folder = './features/'

    opt.model_name = '{}_{}'. \
        format(opt.dataset, opt.model)


    return opt

def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'celeba':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)
    elif opt.dataset == 'celeba':
        train_dataset = datasets.CelebA(root=opt.data_folder,transform=train_transform,target_type=['identity'])
        val_dataset = datasets.CelebA(root=opt.data_folder,split='valid',transform=val_transform,target_type=['identity'])

    else:
        raise ValueError(opt.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

    return train_loader, val_loader

def set_model(opt):
    model = SupConResNet(name=opt.model,head="linear",feat_dim=512)#,head="linear",feat_dim=128)
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        cudnn.benchmark = True
    else:
        model = model.to(device=device)
    model.load_state_dict(state_dict)

    return model


def fe(loader, model, opt):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    features_list = []
    labels_list = []
    end = time.time()
    for idx, (images, labels) in enumerate(loader):
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)

        features_list.append(features.cpu())
        labels_list.append(labels.cpu())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % 10 == 0:
            print('Train: [{0}/{1}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                idx + 1, len(loader), batch_time=batch_time,
                data_time=data_time))
            sys.stdout.flush()

    return torch.cat(features_list), torch.cat(labels_list)

def main():
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model = set_model(opt)


    train_features, train_labels = fe(train_loader,model,opt)
    torch.save(train_features,os.path.join(opt.features_folder,'train_fe.torch'))
    torch.save(train_labels,os.path.join(opt.features_folder,'train_labels.torch'))
    valid_features, valid_labels = fe(val_loader,model,opt)
    torch.save(valid_features,os.path.join(opt.features_folder,'valid_fe.torch'))
    torch.save(valid_labels,os.path.join(opt.features_folder,'valid_labels.torch'))



if __name__ == '__main__':
    main()

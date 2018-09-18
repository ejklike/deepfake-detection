import os
import argparse
from random import shuffle
import torch

from lib.utils import mkdir, get_image_paths_and_labels
from lib.image_loader import ImageLoader
from lib.logger import Logger
from lib.trainer import Trainer
from lib.model_loader import ModelLoader

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Deepfake Detector')

parser.add_argument('-d', '--data-dir', 
                    dest='data_dir', 
                    default='./data',
                    help="input data directory")

parser.add_argument('-mn', '--model-name', 
                    dest='model_name', 
                    default='no_name',
                    help="A model name to be a pth name")

parser.add_argument('--train', 
                    action='store_true', 
                    default=False,
                    help='do training')

parser.add_argument('-b', '--batch-size', 
                    dest='batch_size', 
                    type=int, 
                    default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', 
                    type=int, 
                    default=50, 
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr',
                    type=float, 
                    default=1e-5, 
                    help='learning rate (default: 1e-5)')
parser.add_argument('--no-cuda', 
                    action='store_true', 
                    default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', 
                    type=int, 
                    default=1,
                    help='random seed (default: 1)')


args = parser.parse_args()

# MODEL
Model = ModelLoader().get_model(args.model_name)
model_dir = mkdir(os.path.join('./model_dir', args.model_name))

# TRAINER
trainer = Trainer(
    model=Model,
    model_dir=model_dir, 
    no_cuda=args.no_cuda,
    seed=args.seed,
    lr=args.lr)

# DATALOADERS
print('=> read directory...')
image_paths_and_labels_trn, image_paths_and_labels_tst = \
    get_image_paths_and_labels(args.data_dir, test_ratio=0.2)
print('=> #data (%fake): trn={} ({:.2f}%), tst={} ({:.2f}%)'.format(
    len(image_paths_and_labels_trn), 
    100. * sum([l for p, l in image_paths_and_labels_trn]) / len(image_paths_and_labels_trn),
    len(image_paths_and_labels_tst),
    100. * sum([l for p, l in image_paths_and_labels_tst]) / len(image_paths_and_labels_tst)))

print('=> make dataloaders...')
dataloader_args = {'num_workers': 1, 'pin_memory': True} if args.no_cuda is False else {}

if args.train:
    dataloader = torch.utils.data.DataLoader(
        ImageLoader(image_paths_and_labels_trn), 
        batch_size=args.batch_size, shuffle=True, **dataloader_args)
    logger_trn = Logger(mkdir(os.path.join(model_dir, 'log', 'train')))

testloader = torch.utils.data.DataLoader(
    ImageLoader(image_paths_and_labels_tst), 
    batch_size=500, shuffle=False, **dataloader_args)
logger_tst = Logger(mkdir(os.path.join(model_dir, 'log', 'test')))

if args.train:
    print('\nstart training...\n')
    for _ in range(1, args.epochs + 1):
        trainer.train_one_epoch(dataloader)
        print('==> Test (training set):')
        trainer.test(dataloader, logger_trn)
        print('==> Test (test set):')
        trainer.test(testloader, logger_tst)

trainer.test(testloader, logger_tst, 
    classify_images=True, 
    image_paths_and_labels=image_paths_and_labels_tst)
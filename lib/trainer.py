import os
from shutil import copy

import numpy as np

import torch as t
import torch.nn.functional as F
import torch.nn as nn

from lib.utils import mkdir


def get_optimizer(lr, optimizer_path, parameters):
    optimizer = t.optim.Adam(
        parameters, lr=lr,  betas=(0.5, 0.999))
    if os.path.isfile(optimizer_path):
        optimizer.load_state_dict(t.load(optimizer_path))
    return optimizer


def save_optimizer(optimizer_path, optimizer):
    t.save(optimizer.state_dict(), optimizer_path)


class Trainer:
    def __init__(self, model_dir='./output', model=None,
                 no_cuda=False, seed=1, lr=5e-5):

        # Hyperparameters
        self.model_dir = model_dir
        self.lr = lr
        
        # Torch Seed
        t.manual_seed(seed)

        # CUDA/CUDNN setting
        self.use_cuda = no_cuda is False and t.cuda.is_available()
        self.n_gpu = t.cuda.device_count()
        t.backends.cudnn.benchmark = self.use_cuda
        self.device = t.device("cuda" if self.use_cuda else "cpu")

        # Model
        assert model is not None
        model_path = self.get_path('model.pth')
        optim_path = self.get_path('optim.pth')
        self.model = self.get_model(model, path=model_path)
        self.optim = get_optimizer(self.lr, optim_path, self.model.parameters())

    def get_path(self, name):
        return os.path.join(self.model_dir, name)

    def get_model(self, model_class, **kwargs):
        model = model_class(**kwargs)
        model.load()
        if self.n_gpu > 1:
            model = nn.DataParallel(model)
        return model.to(self.device)
    
    def loss(self, output, target, reduce=None):
        if reduce is None:
            return F.nll_loss(output, target)
        else:
            return F.nll_loss(output, target, reduce=reduce)
    
    def get_epoch(self):
        if self.n_gpu > 1:
            return self.model.module.epoch
        else:
            return self.model.epoch

    def update_epoch(self):
        if self.n_gpu > 1:
            self.model.module.epoch += 1
        else:
            self.model.epoch += 1

    def train_one_epoch(self, dataloader, logger=None):
        self.model.train()
        
        loss_sum = 0.
        for batch_idx, (image, target) in enumerate(dataloader):
            image, target = image.to(self.device), target.to(self.device)
            output = self.model(image)

            self.optim.zero_grad()
            loss = self.loss(output, target)
            loss.backward()
            self.optim.step()

            print('\rEpoch: {}, Loss: {:.6f} [{}/{} ({:.0f}%)]'
                    .format(self.get_epoch() + 1, loss.item(), 
                            batch_idx * len(image), 
                            len(dataloader.dataset), 
                            100. * batch_idx / len(dataloader)), end='')
            loss_sum += loss.item() * len(image)

        self.update_epoch()

        if logger is not None:
            logger.scalar_summary(
                'loss', loss_sum / len(dataloader.dataset), self.get_epoch())

        # save checkpoints
        print('')
        self.save_model()
        save_optimizer(self.get_path('optim.pth'), self.optim)
        print('')


    def save_model(self):
        if self.n_gpu > 1:
            self.model.module.save()
        else:
            self.model.save()


    def test(self, dataloader, logger, 
             classify_images=False, image_paths_and_labels=None):
        self.model.eval()

        test_loss = 0

        correct = 0
        true_positives = 0
        prec_denom = 0
        rec_denom = 0
        
        correct_label = []

        with t.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.loss(output, target).item() * len(data)
                pred = output.max(1)[1] # get the index of the max log-probability

                correct_t = pred.eq(target)
                correct_label.append(correct_t.data.cpu().numpy())
                correct += correct_t.sum().item()

                true_positives += (pred * target).sum().item()
                prec_denom += pred.sum().item()
                rec_denom += target.sum().item()

        test_loss /= len(dataloader.dataset)
        accuracy = correct / len(dataloader.dataset)
        precision = true_positives / prec_denom
        recall = true_positives / rec_denom
        logger.scalar_summary('loss', test_loss, self.get_epoch())
        logger.scalar_summary('accuracy', accuracy, self.get_epoch())
        logger.scalar_summary('precision', precision, self.get_epoch())
        logger.scalar_summary('recall', recall, self.get_epoch())

        print('Loss: {:.4f}, Acc: {}/{} ({:.0f}%), Prec: {}/{} ({:.0f}%), Rec: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(dataloader.dataset), 100. * accuracy,
            true_positives, prec_denom, 100 * precision, 
            true_positives, rec_denom, 100 * recall))

        if classify_images is True:
            assert image_paths_and_labels is not None
            correct_label = np.concatenate(correct_label)
            self._save_misclassified_images(
                correct_label, image_paths_and_labels)

    def _save_misclassified_images(self, correct_label, image_paths_and_labels):
        def _confusion_result(correct, fake):
            if correct == 1:
                if fake == 1: 
                    return 'true_positive'
                else: 
                    return 'true_negative'
            else: # misclassified
                if fake == 1: 
                    return 'false_negative'
                else: 
                    return 'false_positive'

        for correct, (path, fake) in zip(correct_label, image_paths_and_labels):
            target_dir = mkdir(os.path.join(
                self.model_dir, 'test_images', _confusion_result(correct, fake)))
            copy(path, target_dir)
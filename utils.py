import torch
from torch import nn, optim
import torchvision
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import argparse
import glob
import os
import datetime
import time
import sys
import random
from visdom import Visdom



def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x
def denorm(x):
    
    return ((x + 1) / 2).clamp(0, 1)

def load_ckp(checkpoint_fpath, G, F, Dx = None, Dy = None, G_optimizer = None, F_optimizer = None, Dx_optimizer = None, Dy_optimizer = None):
    checkpoint = torch.load(checkpoint_fpath)

    G.load_state_dict(checkpoint['G_model'])
    F.load_state_dict(checkpoint['F_model'])
    if Dx is not None:
        Dx.load_state_dict(checkpoint['Dx_model'])
    if Dy is not None:
        Dy.load_state_dict(checkpoint['Dy_model'])
    
    if G_optimizer is not None:
        G_optimizer.load_state_dict(checkpoint['G_optimizer'])
    if F_optimizer is not None:
        F_optimizer.load_state_dict(checkpoint['F_optimizer'])
    if Dx_optimizer is not None:
        Dx_optimizer.load_state_dict(checkpoint['Dx_optimizer'])
    if Dy_optimizer is not None:
        Dy_optimizer.load_state_dict(checkpoint['Dy_optimizer'])
    epoch = checkpoint['epoch']

    return G, F, Dx, Dy, G_optimizer, F_optimizer, Dx_optimizer, Dy_optimizer, epoch
    
    
def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

class Logger():
    def __init__(self, n_epochs, batches_epoch, image_step, port = 8097):
        self.viz = Visdom(port=port)
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.step = 0
        self.image_step = image_step
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}


    def log(self, losses=None, images=None):
        self.step += 1
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        if (self.step % self.image_step == 0):
            for image_name, tensor in images.items():
                if image_name not in self.image_windows:
                    self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name})
                else:
                    self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), 
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import argparse
import glob
import os
import datetime
import visdom

from utils import load_ckp, to_variable, denorm, Logger
from network import Generator, Discriminator, ResidualBlock


class image_preprocessing(Dataset):
    def __init__(self, root_dir, directory='train'):
        self.root_dir = root_dir
        self.transforms = transforms.Compose([
            transforms.Resize((256, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dir_data = os.path.join(root_dir, directory)
        self.image = glob.glob(os.path.join(root_dir, directory) + '/*.jpg')

    def __getitem__(self, idx):
        AB_path = self.image[idx]
        AB = Image.open(AB_path)

        AB = self.transforms(AB)
        # 3 * 256 * 512
        _, h, w = AB.shape
        A = AB.clone().detach()[:, :, :int(w/2)]
        B = AB.clone().detach()[:, :, int(w/2):]
        return {'A': A, 'B': B}

    def __len__(self):
        return len(self.image)

def patch_loss(criterion, input, TF):
    if TF is True:
        comparison = torch.ones_like(input)
    else:
        comparison = torch.zeros_like(input)
    return criterion(input, comparison)

def least_squares(input, comparision):
    return 0.5 * torch.mean((input - comparision) ** 2)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    start_epoch = 0
    train_image_dataset = image_preprocessing(opt.dataset, 'train')
    data_loader = DataLoader(train_image_dataset, batch_size=opt.batch_size,
                            shuffle=True, num_workers=opt.num_workers)
    criterion = least_squares
    euclidean_l1 = nn.L1Loss()

    G = Generator(ResidualBlock, layer_count=9)
    F = Generator(ResidualBlock, layer_count=9)
    Dx = Discriminator()
    Dy = Discriminator()

    G_optimizer = optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    F_optimizer = optim.Adam(F.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    Dx_optimizer = optim.Adam(Dx.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    Dy_optimizer = optim.Adam(Dy.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

    if torch.cuda.is_available():
        G = nn.DataParallel(G)
        F = nn.DataParallel(F)
        Dx = nn.DataParallel(Dx)
        Dy = nn.DataParallel(Dy)

        G = G.cuda()
        F = F.cuda()
        Dx = Dx.cuda()
        Dy = Dy.cuda()
    
    if opt.checkpoint is not None:
        G, F, Dx, Dy, G_optimizer, F_optimizer, Dx_optimizer, Dy_optimizer, start_epoch = load_ckp(opt.checkpoint, G, F, Dx, Dy, G_optimizer, F_optimizer, Dx_optimizer, Dy_optimizer)

    print('[Start] : Cycle GAN Training')

    logger = Logger(opt.epochs, len(data_loader), image_step=10)

    for epoch in range(opt.epochs):
        epoch = epoch + start_epoch + 1
        print("Epoch[{epoch}] : Start".format(epoch=epoch))
        
        for step, data in enumerate(data_loader):
            real_A = to_variable(data['A'])
            real_B = to_variable(data['B'])

            fake_B = G(real_A)
            fake_A = F(real_B)

            # Train Dx
            Dx_optimizer.zero_grad()

            Dx_real = Dx(real_A)
            Dx_fake = Dx(fake_A)

            Dx_loss = patch_loss(criterion, Dx_real, True) + patch_loss(criterion, Dx_fake, 0)

            Dx_loss.backward(retain_graph=True)
            Dx_optimizer.step()

            # Train Dy
            Dy_optimizer.zero_grad()

            Dy_real = Dy(real_B)
            Dy_fake = Dy(fake_B)

            Dy_loss = patch_loss(criterion, Dy_real, True) + patch_loss(criterion, Dy_fake, 0)

            Dy_loss.backward(retain_graph=True)
            Dy_optimizer.step()

            # Train G
            G_optimizer.zero_grad()

            Dy_fake = Dy(fake_B)

            G_loss = patch_loss(criterion, Dy_fake, True)

            # Train F
            F_optimizer.zero_grad()

            Dx_fake = Dx(fake_A)

            F_loss = patch_loss(criterion, Dx_fake, True)

            # identity loss
            loss_identity = euclidean_l1(real_A, fake_A) + euclidean_l1(real_B, fake_B)

            # cycle consistency
            loss_cycle = euclidean_l1(F(fake_B), real_A) + euclidean_l1(G(fake_A), real_B)

            # Optimize G & F
            loss = G_loss + F_loss + opt.lamda * loss_cycle + opt.lamda * loss_identity * (0.5)

            loss.backward()
            G_optimizer.step()
            F_optimizer.step()

            if (step + 1 ) % opt.save_step == 0:
                print("Epoch[{epoch}]| Step [{now}/{total}]| Dx Loss: {Dx_loss}, Dy_Loss: {Dy_loss}, G_Loss: {G_loss}, F_Loss: {F_loss}".format(
                    epoch=epoch, now=step + 1, total=len(data_loader), Dx_loss=Dx_loss.item(), Dy_loss=Dy_loss,
                    G_loss=G_loss.item(), F_loss=F_loss.item()))
                batch_image = torch.cat((torch.cat((real_A, real_B), 3), torch.cat((fake_A, fake_B), 3)), 2)

                torchvision.utils.save_image(denorm(batch_image[0]), opt.training_result + 'result_{result_name}_ep{epoch}_{step}.jpg'.format(result_name=opt.result_name,epoch=epoch, step=(step + 1) * opt.batch_size))
            
            # http://localhost:8097
            logger.log(
                losses={
                    'loss_G': G_loss,
                    'loss_F': F_loss,
                    'loss_identity': loss_identity,
                    'loss_cycle': loss_cycle,
                    'total_G_loss': loss,
                    'loss_Dx': Dx_loss,
                    'loss_Dy': Dy_loss,
                    'total_D_loss': (Dx_loss + Dy_loss),
                },
                images={
                    'real_A': real_A,
                    'real_B': real_B,
                    'fake_A': fake_A,
                    'fake_ B': fake_B,
                },
            )


        torch.save({
            'epoch': epoch,
            'G_model': G.state_dict(),
            'G_optimizer': G_optimizer.state_dict(),
            'F_model': F.state_dict(),
            'F_optimizer': F_optimizer.state_dict(),
            'Dx_model': Dx.state_dict(),
            'Dx_optimizer': Dx_optimizer.state_dict(),
            'Dy_model': Dy.state_dict(),
            'Dy_optimizer': Dy_optimizer.state_dict(),
        }, opt.save_model + 'model_{result_name}_CycleGAN_ep{epoch}.ckp'.format(result_name=opt.result_name, epoch=epoch))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pytorch CycleGAN implementation')

    parser.add_argument('--dataset', default='./data/datasets/maps/', type=str, help='dataset path')
    parser.add_argument('--save_model', default='./data/', type=str, help='model save directory')
    parser.add_argument('--checkpoint', type=str, help='model checkpoint')
    parser.add_argument('--save_step', default=10, type=int, help='save step')
    parser.add_argument('--training_result', default='./training_result/', type=str, help='training_result')
    parser.add_argument('--result_name', default='maps', type=str, help='model saving name')



    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--epochs', default=10, type=int, help='epochs')
    parser.add_argument('--gpus', default="0", type=str, help='gpus')
    parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.5, type=float, help='beta 1')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta 2')    
    parser.add_argument('--lamda', default=10, type=int, help='lamda')
    parser.add_argument('--num_workers', default=0, type=int, help='number of workers')

    opt = parser.parse_args()

    main()
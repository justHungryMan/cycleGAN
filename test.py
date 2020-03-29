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
from tqdm import tqdm

from utils import load_ckp, to_variable, denorm
from network import Generator, ResidualBlock



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

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    test_image_dataset = image_preprocessing(opt.dataset, 'val')
    data_loader = DataLoader(test_image_dataset, batch_size=opt.batch_size,
                            shuffle=False, num_workers=opt.num_workers)

    G = Generator(ResidualBlock, layer_count=9)
    F = Generator(ResidualBlock, layer_count=9)

    if torch.cuda.is_available():
        G = nn.DataParallel(G)
        F = nn.DataParallel(F)

        G = G.cuda()
        F = F.cuda()

    G, F, _, _, _, _, _, _, _ = load_ckp(opt.model_path, G, F)
    G.eval()
    F.eval()
    
    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)

    for step, data in enumerate(tqdm(data_loader)):
        real_A = to_variable(data['A'])
        real_B = to_variable(data['B'])

        fake_B = G(real_A)
        fake_A = F(real_B)

        batch_image = torch.cat((torch.cat((real_A, real_B), 3), torch.cat((fake_A, fake_B), 3)), 2)
        for i in range(batch_image.shape[0]):
            torchvision.utils.save_image(denorm(batch_image[i]), opt.save_path + '{result_name}_{step}.jpg'.format(result_name=opt.result_name, step=step * opt.batch_size + i))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pytorch CycleGAN implementation')

    parser.add_argument('--dataset', default='./data/datasets/maps/', type=str, help='dataset path')
    parser.add_argument('--save_path', default='./result/', type=str, help='save path')
    parser.add_argument('--model_path', required=True, type=str, help='model path')
    parser.add_argument('--result_name', default='maps', type=str, help='model saving name')

    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--num_workers', default=0, type=int, help='number of workers')
    parser.add_argument('--gpus', default="0", type=str, help='gpus')

    opt = parser.parse_args()

    main()
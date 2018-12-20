import torch
import icdar
import numpy as np
from PIL import Image
import os
from os.path import join as opj
from model import Extractor

from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torchvision import transforms

from tqdm import tqdm
from argparse import ArgumentParser

from icdar import icdar

from CPMMC_python import CPMMC

parser = ArgumentParser()

parser.add_argument("--num_readers", type=int, default=1, help="number of readers for dataloader")
parser.add_argument("--input_size", type=int, default=512, help="image input size of model")
parser.add_argument("--batch_size_per_gpu", type=int, default=16, help="")
parser.add_argument("--gpu_list", type=str, default="0", help="which gpus used for training")
parser.add_argument("--train_path", type=str, default="./data/icdar2015/train")
parser.add_argument("--test_path", type=str, default="./data/icdar2015/test")

parser.add_argument("--icdar_patches", type=str, default="./data/icdar2015/patches")
parser.add_argument("--icdar_true_path", type=str, default="./data/icdar2015/patches/1")
parser.add_argument("--icdar_false_path", type=str, default="./data/icdar2015/patches/0")

args = parser.parse_args()

gpus = list(range(len(args.gpu_list.split(','))))


def make_dataset():

    icdar_loader = icdar(args.train_path)

    pbar = tqdm(total=len(icdar_loader))
    true_count = 0
    false_count = 0

    for cnt, (im, tag) in enumerate(icdar_loader):
        pbar.update(1)
        for count, flag in enumerate(tag):
            if flag:
                true_count += 1
                path = opj(args.icdar_true_path, 'IMG_'+str(true_count)+'.jpg')
                im[count].save(path)
            else:
                false_count += 1
                path = opj(args.icdar_false_path, 'IMG_'+str(false_count)+'.jpg')
                im[count].save(path)

    pbar.close()


def extract(loader, model):
    for cnt, (im, ann) in enumerate(loader):

        feat = model(im)
        import ipdb
        ipdb.set_trace()


if __name__ == '__main__':
    # Step 1 : extract bboxes
    make_dataset()

    # Step 2 : extract features
    transforms = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229,0.224,0.225])])
    dataset = ImageFolder(root=args.icdar_patches, transform=transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

    model = Extractor()
    extract(loader=loader, model=model)

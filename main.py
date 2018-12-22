import torch
import icdar
import numpy as np
from PIL import Image
import os
from os.path import join as opj
from model import Extractor

from torch.autograd import Variable
from torchvision import transforms

from tqdm import tqdm
from argparse import ArgumentParser
import shutil

from icdar import icdar
from folder import ImageFolder
from CPMMC_python import CPMMC

parser = ArgumentParser()

parser.add_argument("--num_readers", type=int, default=1, help="number of readers for dataloader")
parser.add_argument("--input_size", type=int, default=512, help="image input size of model")
parser.add_argument("--batch_size_per_gpu", type=int, default=16, help="")
parser.add_argument("--gpu_list", type=str, default="0", help="which gpus used for training")
parser.add_argument("--train_path", type=str, default="./data/icdar2015/train")
parser.add_argument("--test_path", type=str, default="./data/icdar2015/test")

# default training path
parser.add_argument("--icdar_patches", type=str, default="./data/icdar2015/patches")
parser.add_argument("--icdar_true_path", type=str, default="./data/icdar2015/patches/1")
parser.add_argument("--icdar_false_path", type=str, default="./data/icdar2015/patches/0")
# testing path
parser.add_argument("--icdar_patches_test", type=str, default="./data/icdar2015/test_patches")
parser.add_argument("--icdar_true_path_test", type=str, default="./data/icdar2015/test_patches/1")
parser.add_argument("--icdar_false_path_test", type=str, default="./data/icdar2015/test_patches/0")


parser.add_argument("--split_path_0", type=str, default="./tmp/result/0")
parser.add_argument("--split_path_1", type=str, default="./tmp/result/1")

args = parser.parse_args()

gpus = list(range(len(args.gpu_list.split(','))))


def make_dataset(dataset):

    if dataset == 'train':
        root = args.train_path
        split_true_path = args.icdar_true_path
        split_false_path = args.icdar_false_path
    else:
        root = args.test_path
        split_true_path = args.icdar_true_path_test
        split_false_path = args.icdar_false_path_test

    icdar_loader = icdar(root)

    pbar = tqdm(total=len(icdar_loader))
    true_count = 0
    false_count = 0

    for cnt, (im, tag) in enumerate(icdar_loader):
        pbar.update(1)
        for count, flag in enumerate(tag):
            if flag:
                true_count += 1
                path = opj(split_true_path, 'IMG_'+str(true_count)+'.jpg')
                im[count].save(path)
            else:
                false_count += 1
                path = opj(split_false_path, 'IMG_'+str(false_count)+'.jpg')
                im[count].save(path)

    pbar.close()


def clustering(loader, model):

    print("=======> clustering")
    pbar = tqdm(total=len(loader))

    #convert into (1, 1) numpy array for further deployment
    C = np.array([[0.01]])
    l = np.array([[10]])
    b_0 = np.array([[0]])
    xi_0 = np.array([[0.5]])

    epsilon = 0.1

    MMC = CPMMC.CPMMC(dim=2048, C=C, \
                      epsilon=epsilon, \
                      l=l,b_0=b_0,xi_0=xi_0)

    cls0_path = args.split_path_0
    cls1_path = args.split_path_1

    model = model.cuda()

    for cnt, (data, label, path) in enumerate(loader):

        pbar.update(1)

        data = data.cuda()
        feat = model(data)
        feat = feat.cpu()

        data = feat.numpy()
        new_label = -np.ones(label.shape, dtype=np.int8)
        for count, num in enumerate(label):
            if num == 1:
                new_label[count] = 1

        acc, pred = MMC(data=data, label=new_label)

        import ipdb
        ipdb.set_trace()
        for cnt, flag in enumerate(pred):
            if flag.item() == 1:
                shutil.copy(path[cnt], cls0_path)
            elif flag.item() == -1:
                shutil.copy(path[cnt], cls1_path)

    pbar.close()
    return acc


def filter():
    transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229,0.224,0.225])])
    dataset = ImageFolder(root=args.icdar_patches, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

    model = Extractor()
    for params in model.parameters():
        params.requires_grad = False
    acc = clustering(loader=loader, model=model)

if __name__ == '__main__':
    # Step 1 : extract bboxes
    dataset = ['train', 'test']
    make_dataset(dataset=dataset[1])

    # Step 2 : extract features and clustering
    """
    transforms = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229,0.224,0.225])])
    dataset = ImageFolder(root=args.icdar_patches, transform=transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

    model = Extractor()
    for params in model.parameters():
        params.requires_grad = False
    acc = clustering(loader=loader, model=model)
    """
    #filter()
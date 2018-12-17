import torch
import icdar
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from model import Extractor

from torch.autograd import Variable

from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--num_readers", type=int, default=1, help="number of readers for dataloader")
parser.add_argument("--input_size", type=int, default=512, help="image input size of model")
parser.add_argument("--batch_size_per_gpu", type=int, default=16, help="")
parser.add_argument("--gpu_list", type=str, default="0", help="which gpus used for training")

args = parser.parse_args()

"""
tf.app.args.DEFINE_integer("num_readers", 1, "")
tf.app.args.DEFINE_integer("input_size", 512, "")
tf.app.args.DEFINE_integer("batch_size_per_gpu", 16, "")
tf.app.args.DEFINE_string("gpu_list", "0", "")

args = tf.app.args.args
"""

gpus = list(range(len(args.gpu_list.split(','))))

def main():

    batch_size = args.batch_size_per_gpu * len(gpus)
    data_generator = icdar.get_batch(num_workers=args.num_readers,
                                     input_size=args.input_size,
                                     batch_size=args.batch_size_per_gpu * len(gpus))

    model = Extractor()

    ## FIXME test
    iter = 10
    tmp_path = './tmp'
    pbar = tqdm(total=iter)

    feat_path = './features'

    for i in range(iter):
        """
        data: 
            index 0: input images 
            index 1: input images path
            index 2: input score maps
            index 3: input geo maps
            index 4: input training masks
        """
        pbar.update(1)
        data = next(data_generator)
        pre_img = []
        for j in range(batch_size):
            pre_img.append(data[0][j].transpose(2,0,1)[np.newaxis,:] / 255.0)

        imgs = np.concatenate(pre_img)
        im = Variable(torch.from_numpy(imgs))

        feat_path  = os.path.join(feat_path, os.path.basename(data[1][0]).split('.')[0], '.npy')
        idx_path  = os.path.join(feat_path, os.path.basename(data[1][0]).split('.')[0], '.idx')
        # feat = model(im)

        """
        for j in range(batch_size):
            import ipdb
            ipdb.set_trace()
            image_path = os.path.join(tmp_path, 'img', os.path.basename(data[1][j]))
            #score_path = os.path.join(tmp_path, 'score', os.path.basename(data[1][j]))
            mask_path = os.path.join(tmp_path, 'mask', os.path.basename(data[1][j]))
            # im = Image.fromarray(data[0][j].astype('uint8'))
            # score_map = Image.fromarray(data[2][j][:,:,0].astype('uint8'))
            mask_map = Image.fromarray(data[4][j][:,:,0].astype('uint8'))
            # im.save(image_path)
            # score_map.save(score_path)
            mask_map.save(mask_path)
        """

    pbar.close()

if __name__ == '__main__':
    main()
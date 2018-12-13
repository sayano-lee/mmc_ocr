import icdar
import tensorflow as tf
import numpy as np
from PIL import Image
import os

from tqdm import tqdm


tf.app.flags.DEFINE_integer("num_readers", 1, "")
tf.app.flags.DEFINE_integer("input_size", 512, "")
tf.app.flags.DEFINE_integer("batch_size_per_gpu", 14, "")
tf.app.flags.DEFINE_string("gpu_list", "0", "")

FLAGS = tf.app.flags.FLAGS

gpus = list(range(len(FLAGS.gpu_list.split(','))))

def main():

    batch_size = FLAGS.batch_size_per_gpu * len(gpus)
    data_generator = icdar.get_batch(num_workers=FLAGS.num_readers,
                                     input_size=FLAGS.input_size,
                                     batch_size=FLAGS.batch_size_per_gpu * len(gpus))


    ## FIXME test
    iter = 100
    tmp_path = './tmp'
    pbar = tqdm(total=iter)

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
        for j in range(batch_size):
            image_path = os.path.join(tmp_path, 'img', os.path.basename(data[1][j]))
            score_path = os.path.join(tmp_path, 'score', os.path.basename(data[1][j]))
            im = Image.fromarray(data[0][j].astype('uint8'))
            score_map = Image.fromarray(data[2][j][:,:,0].astype('uint8'))
            im.save(image_path)
            score_map.save(score_path)

        # import ipdb
        # ipdb.set_trace()
    pbar.close()

if __name__ == '__main__':
    main()
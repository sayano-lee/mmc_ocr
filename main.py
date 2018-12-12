import icdar
import tensorflow as tf
import numpy as np
from PIL import Image
import os


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

    for i in range(iter):
        """
        data: 
            index 0: 
            index 1:
            index 2:
            index 3:
            index 4:
        """
        data = next(data_generator)
        for j in range(batch_size):
            import ipdb
            ipdb.set_trace()
            image_path = os.path.join(tmp_path, data[1][j])
            Image.fromarray(data[j].astype('uint8'))


        import ipdb
        ipdb.set_trace()

if __name__ == '__main__':
    main()
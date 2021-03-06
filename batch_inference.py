import argparse
import os
import tensorflow as tf
import numpy as np
import glob
from tqdm import tqdm
from deeplab_resnet import DeepLabResNetModel, dense_crf, inv_preprocess, prepare_label, decode_labels, threshold
from PIL import Image

# tuning constants
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
INPUT_DIRECTORY = './dataset/VOCdevkit'
DATA_LIST_PATH = './dataset/val.txt'
OUTPUT_DIRECTORY = './output'
IGNORE_LABEL = 255
NUM_CLASSES = 21
RESTORE_FROM = './deeplab_resnet.ckpt'
RESIZE_TO = [512, 512]


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Example command: \n python .\\batch_inference.py --input-dir=./dataset/jenn_images --output-dir=./output --restore-from=deeplab_resnet.ckpt")
    parser.add_argument("--input-dir", type=str, default=INPUT_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--augment", action='store_true',
                        help="Use prediction-time augemntation, predict output of 4 rotations and average.")
    parser.add_argument("--crf", action='store_true',
                        help="Use a CRF to clean up prediction.")
    parser.add_argument('--thresh', nargs='+', type=float, default=None,
                        help='a certain class and probability threshold at which assign that class')
    return parser.parse_args()


def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


class InferenceImageReader(object):
    def __init__(self, data_dir, img_mean, coord, resize_to):
        self.data_dir = data_dir
        self.coord = coord
        self.image_list = glob.glob(os.path.join(data_dir, "*"))
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images])  # not shuffling if it is val
        img_contents = tf.read_file(self.queue[0])
        img = tf.image.resize_images(tf.image.decode_png(img_contents, channels=3), resize_to)
        img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
        self.image = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
        # Extract mean.
        self.image -= img_mean


if __name__ == '__main__':
    args, preds = get_arguments(), []

    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = InferenceImageReader(
            args.input_dir,
            IMG_MEAN,
            coord,
            RESIZE_TO)
        image_orig = reader.image

    for rots in range(4):
        image = tf.image.rot90(image_orig, k=rots)
        image_batch = tf.expand_dims(image, dim=0)

        # Create network.
        net = DeepLabResNetModel({'data': image_batch}, is_training=False, num_classes=args.num_classes)
        tf.get_variable_scope().reuse_variables()

        # Which variables to load.
        restore_var = tf.global_variables()

        # Predictions.
        raw_output = net.layers['fc1_voc12']
        raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])

        # CRF.
        if args.crf:
            inv_image = tf.py_func(inv_preprocess, [image_batch, 1, IMG_MEAN], tf.uint8)
            raw_output = tf.py_func(dense_crf, [tf.nn.softmax(raw_output), inv_image], tf.float32)

        # Rotate to original
        raw_output = tf.image.rot90(tf.squeeze(raw_output), k=(4-rots))
        raw_output = tf.expand_dims(raw_output, dim=0)
        preds.append(raw_output)

        if not args.augment:
            break

    pred = tf.reduce_mean(tf.concat(preds, axis=0), axis=0)

    # Set class based on threshold
    if args.thresh:
        pred = tf.py_func(threshold, [tf.nn.softmax(pred), int(args.thresh[0]), float(args.thresh[1])], tf.int32)
        pred = tf.expand_dims(pred, dim=0)
    else:
        pred = tf.argmax(tf.expand_dims(pred, dim=0), dimension=3)

    pred = tf.cast(tf.expand_dims(pred, dim=3), tf.int32) # create 4D tensor

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)
    sess.run(tf.local_variables_initializer())

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if args.restore_from is not None:
        load(loader, sess, args.restore_from)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # make output dir if
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # Iterate over training steps.
    for step in tqdm(range(len(reader.image_list))):
        preds = sess.run(pred)
        msk = decode_labels(preds, num_classes=args.num_classes)
        im = Image.fromarray(msk[0])
        im.save(os.path.join(args.output_dir, os.path.basename(reader.image_list[step])+'_mask.png'))
    coord.request_stop()
    coord.join(threads)

"""Evaluation script for the DeepLab-ResNet network on the validation subset
   of PASCAL VOC dataset.

This script evaluates the model on 1449 validation images.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

from PIL import Image
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, dense_crf, inv_preprocess, prepare_label

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = './../footprints/'
TARGET_LIST_PATH = './../footprints/splits/target.txt'
IGNORE_LABEL = 128
NUM_CLASSES = 2
NUM_STEPS = 4488 # Number of images in the target set.
RESTORE_FROM = './models/9-14-17/train_1/model_0.491834_viou.ckpt-22500'
SAVE_DIR = './target-preds/'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--target-list", type=str, default=TARGET_LIST_PATH,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during prediction.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of images in the target set.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted masks.")
    parser.add_argument("--augment", action='store_true',
                        help="Use prediction-time augemntation, predict output of 4 rotations and average.")
    parser.add_argument("--crf", action='store_true',
                        help="Use a CRF to clean up prediction.")
    parser.add_argument("--heatmap", type=int, default='-1',
                        help="createa heatmap of likelihood for the class specified.")
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

def main():
    args, preds = get_arguments(), []

    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.target_list,
            None, # No defined input size.
            False, # No random scale.
            False, # No random mirror.
            args.ignore_label,
            IMG_MEAN,
            coord)
        image_orig = reader.image

    for rots in range(4):
        image = tf.image.rot90(image_orig, k=rots)
        image_batch = tf.expand_dims(image, dim=0) # Add one batch dimension.

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

    if args.heatmap < 0:
        pred = tf.argmax(tf.expand_dims(pred, dim=0), dimension=3)
        pred = tf.cast(tf.expand_dims(pred, dim=3), tf.int32)
    else:
        pred = tf.expand_dims(pred[:,:,args.heatmap], dim=0)
        pred = tf.cast(pred, tf.int32)

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)
    sess.run(tf.local_variables_initializer())

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, args.restore_from)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Iterate over training steps.
    for step in tqdm(range(args.num_steps)):
        preds, img_path = sess.run([pred, reader.queue[0]])

        if args.heatmap < 0:
            preds = decode_labels(preds, num_classes=args.num_classes)
            im = Image.fromarray(preds[0])
        else:
            pr = np.zeros((1,preds.shape[1],preds.shape[2],3))
            preds += abs(np.min(preds))
            preds *= 255/np.max(preds)
            pr[:,:,:,0] = preds
            pr[:,:,:,1] = preds
            pr[:,:,:,2] = preds
            im = Image.fromarray(pr[0].astype('uint8'))

        img_name = os.path.split(img_path)[-1]
        im.save(os.path.join(args.save_dir + img_name))
    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()

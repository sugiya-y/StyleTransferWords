#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generates stylized images with different strengths of a stylization.

For each pair of the content and style images this script computes stylized
images with different strengths of stylization (interpolates between the
identity transform parameters and the style parameters for the style image) and
saves them to the given output_dir.
See run_interpolation_with_identity.sh for example usage.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import os
import time
import cv2

import numpy as np
import tensorflow as tf

import arbitrary_image_stylization_build_model as build_model
from magenta.models.image_stylization import image_utils
# from tinynet import wordQueryNet
# from chainer import cuda, Variable, optimizers, serializers
# from tinynet import wordQueryNet
# from wordparam import word2vector
# from vggparam import vggparamater
# from vggnet import VGGNet
# from chainer import cuda
import pickle
# import chainer

slim = tf.contrib.slim

flags = tf.flags
flags.DEFINE_string('checkpoint', 'arbitrary_style_transfer/model.ckpt', 'Path to the model checkpoint.')
flags.DEFINE_string('style_images_paths', 'images/valid/*.jpg', 'Paths to the style images'
                    'for evaluation.')
flags.DEFINE_string('content_images_paths', 'images/valid/*.jpg', 'Paths to the content images'
                    'for evaluation.')
flags.DEFINE_string('output_dir', 'out_ours', 'Output directory.')
flags.DEFINE_integer('image_size', 256, 'Image size.')
flags.DEFINE_boolean('content_square_crop', False, 'Wheather to center crop'
                     'the content image to be a square or not.')
flags.DEFINE_integer('style_image_size', 256, 'Style image size.')
flags.DEFINE_boolean('style_square_crop', False, 'Wheather to center crop'
                     'the style image to be a square or not.')
flags.DEFINE_integer('maximum_styles_to_evaluate', 1024, 'Maximum number of'
                     'styles to evaluate.')
flags.DEFINE_string('interpolation_weights', '[1.0]', 'List of weights'
                    'for interpolation between the parameters of the identity'
                    'transform and the style parameters of the style image. The'
                    'larger the weight is the strength of stylization is more.'
                    'Weight of 1.0 means the normal style transfer and weight'
                    'of 0.0 means identity transform.')
flags.DEFINE_boolean('color_preserve', False, 'boolean coloer preserve mode')
FLAGS = flags.FLAGS


def main(unused_argv=None):
    print('timer start')
    start = time.time()

    words = ['布', '植物', 'ガラス', '革', '金属', '紙', 'プラスチック', '石', '水', '木', '樹脂', 'アクリル', 'アルミニウム', '牛皮', 'レンガ', '絹']

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)

    tf.logging.set_verbosity(tf.logging.INFO)
    if not tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.MkDir(FLAGS.output_dir)

    with tf.Graph().as_default(), sess:
        # Defines place holder for the style image.
        style_img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])
        if FLAGS.style_square_crop:
            style_img_preprocessed = image_utils.center_crop_resize_image(
                style_img_ph, FLAGS.style_image_size)
        else:
            style_img_preprocessed = image_utils.resize_image(style_img_ph,
                                                              FLAGS.style_image_size)

        # Defines place holder for the content image.
        content_img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])
        if FLAGS.content_square_crop:
            content_img_preprocessed = image_utils.center_crop_resize_image(
                content_img_ph, FLAGS.image_size)
        else:
            content_img_preprocessed = image_utils.resize_image(
                content_img_ph, FLAGS.image_size)

        # Defines the model.
        stylized_images, _, _, bottleneck_feat = build_model.build_model(
            content_img_preprocessed,
            style_img_preprocessed,
            trainable=False,
            is_training=False,
            inception_end_point='Mixed_6e',
            style_prediction_bottleneck=100,
            adds_losses=False)

        if tf.gfile.IsDirectory(FLAGS.checkpoint):
            checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
        else:
            checkpoint = FLAGS.checkpoint
            tf.logging.info(
                'loading latest checkpoint file: {}'.format(checkpoint))

        init_fn = slim.assign_from_checkpoint_fn(checkpoint,
                                                 slim.get_variables_to_restore())
        sess.run([tf.local_variables_initializer()])
        init_fn(sess)

        # Gets the list of the input style images.
        style_img_list = tf.gfile.Glob(FLAGS.style_images_paths)
        if len(style_img_list) > FLAGS.maximum_styles_to_evaluate:
            np.random.seed(1234)
            style_img_list = np.random.permutation(style_img_list)
            style_img_list = style_img_list[:FLAGS.maximum_styles_to_evaluate]

        # Gets list of input content images.
        content_img_list = tf.gfile.Glob(FLAGS.content_images_paths)

        j = -1
        for content_i, content_img_path in enumerate(content_img_list):
            j += 1
            content_img_np = image_utils.load_np_image_uint8(content_img_path)[:, :, :
                                                                               3]
            content_img_name = os.path.basename(content_img_path)[:-4]

            # Saves preprocessed content image.
            inp_img_croped_resized_np = sess.run(
                content_img_preprocessed, feed_dict={
                    content_img_ph: content_img_np
                })
            image_utils.save_np_image(inp_img_croped_resized_np,
                                      os.path.join(FLAGS.output_dir,
                                                   '%s.jpg' % (content_img_name)))

            if FLAGS.color_preserve is True:
                print('color preserve mode!')
                # convert content iamge to ycc from bgr
                height, width, channels = inp_img_croped_resized_np[0].shape[:3]
                # print(inp_img_croped_resized_np)
                wgap = 4 - (width % 4)  # fuking translater made some gaps because of decoder
                hgap = 4 - (height % 4)
                inp_img_croped_resized_np = inp_img_croped_resized_np * 255
                content_img_np_ycc = cv2.cvtColor(inp_img_croped_resized_np[0], cv2.COLOR_RGB2YCR_CB)
                # print(content_img_np_ycc)
                zeros = np.zeros((height, width), content_img_np_ycc.dtype)
                zeros = zeros + 128  # YCC's zero is center of 255
                tmp = cv2.cvtColor(content_img_np_ycc, cv2.COLOR_YCR_CB2BGR)
                cv2.imwrite("gray.jpg", tmp)
                # print(zeros)
                Ycontent, Crcontent, Cbcontent = cv2.split(content_img_np_ycc)
                # print(Ycontent.shape, Crcontent.shape, Cbcontent.shape, zeros.shape)
                # print(Crcontent)
                # print(content_img_np_ycc)
                # content_img_np_ycc_y = cv2.merge((Y, zeros, zeros))
                # content_img_np_gry = cv2.cvtColor(content_img_np_ycc_y, cv2.COLOR_YCR_CB2RGB)
                # print(content_img_np_gry)
                # cv2.imwrite("gray.jpg", content_img_np_gry)
                # print(np.shape(content_img_np))
                # content_img_np = content_img_np_gry

            # Computes bottleneck features of the style prediction network for the
            # identity transform.

            identity_params = sess.run(
                bottleneck_feat, feed_dict={style_img_ph: content_img_np})

            i = 0
            for word in words:
                # word = words[i]
                print(word)
                i += 1
                # if style_i > FLAGS.maximum_styles_to_evaluate:
                    # break
                # style_img_name = os.path.basename(style_img_path)[:-4]
                # style_image_np = image_utils.load_np_image_uint8(style_img_path)[:, :, :
                                                                                 # 3]

                # if style_i % 10 == 0:
                    # tf.logging.info('Stylizing (%d) %s with (%d) %s' %
                                    # (content_i, content_img_name, style_i,
                                     # style_img_name))

                # Saves preprocessed style image.
                # style_img_croped_resized_np = sess.run(
                    # style_img_preprocessed, feed_dict={
                        # style_img_ph: style_image_np
                    # })
                # image_utils.save_np_image(style_img_croped_resized_np,
                                          # os.path.join(FLAGS.output_dir,
                                                       # '%s.jpg' % (style_img_name)))

                # Computes bottleneck features of the style prediction network for the
                # given style image.
                # style_params_ori = sess.run(
                    # bottleneck_feat, feed_dict={style_img_ph: style_image_np})

                # print(np.shape(style_params))
                picklename = 'params/{}_{}.pickle'.format(word, j)
                f = open(picklename, 'r')
                style_params = pickle.load(f)
                # print(style_params)

                # print('diff of original para and made para:')
                # print(style_params_ori - style_params)

                interpolation_weights = ast.literal_eval(
                    FLAGS.interpolation_weights)
                # Interpolates between the parameters of the identity transform and
                # style parameters of the given style image.
                for interp_i, wi in enumerate(interpolation_weights):
                    stylized_image_res = sess.run(
                        stylized_images,
                        feed_dict={
                            bottleneck_feat:
                                identity_params * (1 - wi) + style_params * wi,
                            content_img_ph:
                                content_img_np
                        })

                    if FLAGS.color_preserve is True:
                        # print(stylized_image_res[0].shape)
                        stylized_image_res_ycc = cv2.cvtColor(stylized_image_res[0], cv2.COLOR_RGB2YCR_CB)
                        Ystylized, Crstylized, Cbstylized = cv2.split(stylized_image_res_ycc)
                        if wgap == 4:  # if original image is just fit
                            Ystylized_crop = Ystylized * 255
                        else:
                            Ystylized_crop = Ystylized[:, :-1 * wgap] * 255
                        if hgap == 4:
                            Ystylized_crop = Ystylized_crop
                        else:
                            Ystylized_crop = Ystylized_crop[:-1 * hgap, :]
                        print(Ystylized_crop.shape, Cbcontent.shape)
                        # print(wgap)
                        swapped_ycc = cv2.merge((Ystylized_crop, Crcontent, Cbcontent))
                        # print(swapped_ycc)
                        stylized_image_res = cv2.cvtColor(swapped_ycc, cv2.COLOR_YCR_CB2BGR)
                        # print(stylized_image_res)
                        cv2.imwrite(os.path.join(FLAGS.output_dir, '%s_stylized_%s_%d.jpg' % (content_img_name, word, interp_i)), stylized_image_res)

                    # Saves stylized image.
                    else:
                        image_utils.save_np_image(
                            stylized_image_res,
                            os.path.join(FLAGS.output_dir, '%s_stylized_%s_%d.jpg' %
                                         (content_img_name, word, interp_i)))

    elapsed_time = time.time() - start
    print("timer stop")
    print(elapsed_time)


if __name__ == '__main__':
    tf.app.run(main)

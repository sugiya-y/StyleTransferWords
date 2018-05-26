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

import os

import numpy as np
import tensorflow as tf

import arbitrary_image_stylization_build_model as build_model
from magenta.models.image_stylization import image_utils

slim = tf.contrib.slim
checkpoint_in = 'arbitrary_style_transfer/model.ckpt'
output_dir = 'outputs_style'
image_size = 256
content_square_crop = False
style_image_size = 256
style_square_crop = False
maximum_styles_to_evaluate = 1024
interpolation_weights_in = '[1.0]'


def styleParam(style_images_paths):
    tf.logging.set_verbosity(tf.logging.INFO)
    style_param_matrix = []
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MkDir(output_dir)

    with tf.Graph().as_default(), tf.Session() as sess:
        # Defines place holder for the style image.
        style_img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])
        if style_square_crop:
            style_img_preprocessed = image_utils.center_crop_resize_image(
                style_img_ph, style_image_size)
        else:
            style_img_preprocessed = image_utils.resize_image(style_img_ph,
                                                              style_image_size)

        # Defines place holder for the content image.
        content_img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])
        if content_square_crop:
            content_img_preprocessed = image_utils.center_crop_resize_image(
                content_img_ph, image_size)
        else:
            content_img_preprocessed = image_utils.resize_image(
                content_img_ph, image_size)

        # Defines the model.
        stylized_images, _, _, bottleneck_feat = build_model.build_model(
            content_img_preprocessed,
            style_img_preprocessed,
            trainable=False,
            is_training=False,
            inception_end_point='Mixed_6e',
            style_prediction_bottleneck=100,
            adds_losses=False)

        if tf.gfile.IsDirectory(checkpoint_in):
            checkpoint = tf.train.latest_checkpoint(checkpoint_in)
        else:
            checkpoint = checkpoint_in
            tf.logging.info(
                'loading latest checkpoint file: {}'.format(checkpoint))

        init_fn = slim.assign_from_checkpoint_fn(checkpoint,
                                                 slim.get_variables_to_restore())
        sess.run([tf.local_variables_initializer()])
        init_fn(sess)

        # Gets the list of the input style images.
        style_img_list = tf.gfile.Glob(style_images_paths)
        if len(style_img_list) > maximum_styles_to_evaluate:
            np.random.seed(1234)
            style_img_list = np.random.permutation(style_img_list)
            style_img_list = style_img_list[:maximum_styles_to_evaluate]

        for style_i, style_img_path in enumerate(style_img_list):
            if style_i > maximum_styles_to_evaluate:
                break
            style_img_name = os.path.basename(style_img_path)[:-4]
            style_image_np = image_utils.load_np_image_uint8(style_img_path)[:, :, :
                                                                             3]
            # Saves preprocessed style image.
            style_img_croped_resized_np = sess.run(
                style_img_preprocessed, feed_dict={
                    style_img_ph: style_image_np
                })
            image_utils.save_np_image(style_img_croped_resized_np,
                                      os.path.join(output_dir,
                                                   '%s.jpg' % (style_img_name)))

            # Computes bottleneck features of the style prediction network for the
            # given style image.
            style_params = sess.run(
                bottleneck_feat, feed_dict={style_img_ph: style_image_np})
            style_param_matrix.append(style_params)

    # style_param_matrix is (num_of_images,1,1,100) vector
    return(style_param_matrix)

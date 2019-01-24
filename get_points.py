from __future__ import print_function, unicode_literals

import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from hand3d.utils.general import *    
from hand3d.nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from io import BytesIO


def create_session():
    # network input
    image_tf = tf.placeholder(tf.float32, shape=(1, 240, 320, 3))
    hand_side_tf = tf.constant([[1.0, 1.0]])  # left hand (true for all samples provided)
    evaluation = tf.placeholder_with_default(True, shape=())

    # build network
    net = ColorHandPose3DNetwork()
    hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,\
    keypoints_scoremap_tf, keypoint_coord3d_tf = net.inference(image_tf, hand_side_tf, evaluation)

    # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # initialize network
    net.init(sess)
    return sess, hand_scoremap_tf, image_crop_tf, scale_tf, center_tf, keypoints_scoremap_tf, keypoint_coord3d_tf, image_tf

def get_handpoints(image_raw, sess, hand_scoremap_tf, image_crop_tf, scale_tf, center_tf, keypoints_scoremap_tf, keypoint_coord3d_tf, image_tf):
    # image_raw = scipy.misc.imread(img_name)
    # image_raw = np.fromstring(image_raw, dtype=np.float64)
    # print("SHAPE ----- ", image_raw.shape)
    image_raw = scipy.misc.imresize(image_raw, (240, 320))
    image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)

    hand_scoremap_v, image_crop_v, scale_v, center_v,\
    keypoints_scoremap_v, keypoint_coord3d_v = sess.run([hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,
                                                            keypoints_scoremap_tf, keypoint_coord3d_tf],
                                                        feed_dict={image_tf: image_v})

    hand_scoremap_v = np.squeeze(hand_scoremap_v)
    image_crop_v = np.squeeze(image_crop_v)
    keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
    keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)

    # post processing
    image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
    coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
    coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)

    # visualize
    fig = plt.figure(1)
    ax1 = fig.add_subplot(221)
    # ax2 = fig.add_subplot(222)
    # ax3 = fig.add_subplot(223)
    # ax4 = fig.add_subplot(224, projection='3d')
    ax1.imshow(image_raw)
    # print("COORD---------" , coord_hw)
    hand_map = (np.argmax(hand_scoremap_v, 2))
    reference_map = np.zeros((hand_map.shape))
    plt.axis('off')
    # fig.axes.get_xaxis().set_visible(False)
    # fig.axes.get_yaxis().set_visible(False)

    if (reference_map == hand_map).all():
        pass
    else:
        plot_hand(coord_hw, ax1)

    img = BytesIO()
    fig.savefig(img, bbox_inches=0, pad_inches = 0)
    img.seek(0)
    fig.clf()
    return img.read()
    # ax2.imshow(image_crop_v)
    # plot_hand(coord_hw_crop, ax2)
    # ax3.imshow(np.argmax(hand_scoremap_v, 2))
    # plot_hand_3d(keypoint_coord3d_v, ax4)
    # ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
    # ax4.set_xlim([-3, 3])
    # ax4.set_ylim([-3, 1])
    # ax4.set_zlim([-3, 3])
    # plt.show()

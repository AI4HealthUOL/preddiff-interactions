"""
code by Jialun Peng, Dong Liu, Songcen Xu, Houqiang Li for

Generating Diverse Structure for Image Inpainting With Hierarchical VQ-VAE

adapted for tensorflow 2
"""
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
#from tensorflow.contrib.framework.python.ops import arg_scope
from . import nn

def texture_generator_spec(x, mask, s, nr_channel=64):
    """
    Input:
    Tensor x of shape (N,H,W,3) (e.g. (8,256,256,3))
    Tensor mask of shape (N,H,W,1) (e.g. (8,256,256,1))
    Tensor s of shape (N,H//8,W//8,C) (e.g. (8,32,32,64))
    Output:
    Tensor x_out of shape (N,H,W,3) (e.g. (8,256,256,3))
    """

    counters = {}
    #with arg_scope([nn.conv2d, nn.gated_conv2d, nn.gated_deconv2d], counters=counters):
    ones_x = tf.ones_like(x)[:, :, :, 0:1]
    x_in = tf.concat([x, ones_x, ones_x*mask], axis=3)

    cnum = nr_channel

    # Encoder
    pl1, counters = nn.gated_conv2d(x_in, cnum, [5,5], [1,1], counters=counters)
    pl1, counters = nn.gated_conv2d(pl1, cnum*2, [3,3], [2,2], counters=counters)
    pl1, counters = nn.gated_conv2d(pl1, cnum*2, [3,3], [1,1], counters=counters)
    pl2, counters = nn.gated_conv2d(pl1, cnum*4, [3,3], [2,2], counters=counters)
    # Upsample structure feature maps (with quantization)
    x_s, counters = nn.gated_conv2d(s, cnum*4, [3,3], [1,1], counters=counters)
    x_s, counters = nn.gated_conv2d(x_s, cnum*4, [3,3], [1,1], counters=counters)
    x_s, counters = nn.gated_deconv2d(x_s, cnum*4, counters=counters)
    pl2 = tf.concat([pl2, x_s], axis=-1)
    pl2, counters = nn.gated_conv2d(pl2, cnum*4, [3,3], [1,1], counters=counters)
    pl2, counters = nn.gated_conv2d(pl2, cnum*4, [3,3], [1,1], counters=counters)
    pl2, counters = nn.gated_conv2d(pl2, cnum*4, [3,3], [1,1], rate=2, counters=counters)
    pl2, counters = nn.gated_conv2d(pl2, cnum*4, [3,3], [1,1], rate=4, counters=counters)
    pl2, counters = nn.gated_conv2d(pl2, cnum*4, [3,3], [1,1], rate=8, counters=counters)
    pl2, counters = nn.gated_conv2d(pl2, cnum*4, [3,3], [1,1], rate=16, counters=counters)

    # Attention transfer under the guidance of structure feature maps
    pl1_att, pl2_att = nn.attention_transfer(s, pl1, pl2, 3, 1, 3, softmax_scale=50., fuse=True)

    # Decoder
    x, counters = nn.gated_conv2d(pl2, cnum*4, [3,3], [1,1], counters=counters)
    x, counters = nn.gated_conv2d(x, cnum*4, [3,3], [1,1], counters=counters)
    x = tf.concat([x, pl2_att], axis=-1)
    x, counters = nn.gated_conv2d(x, cnum*4, [3,3], [1,1], counters=counters)
    x, counters = nn.gated_deconv2d(x, cnum*2, counters=counters)
    x = tf.concat([x, pl1_att], axis=-1)
    x, counters = nn.gated_conv2d(x, cnum*2, [3,3], [1,1], counters=counters)
    x, counters = nn.gated_deconv2d(x, cnum, counters=counters)
    x, counters = nn.gated_conv2d(x, cnum, [3,3], [1,1], counters=counters)
    x, counters = nn.conv2d(x, 3, [3,3], [1,1], nonlinearity=tf.nn.tanh, counters=counters)
    x_out = x

    return x_out        

def texture_discriminator_spec(x, nr_channel=64):
    """
    Input:
    Tensor x of shape (2*N,H,W,4) (e.g. (16,256,256,4))
    Output:
    Tensor x_out of shape (2*N,(H/64)*(W/64)*C") (e.g. (16,4*4*256))
    """
    counters = {}
    #with arg_scope([nn.snconv2d], filter_size=[5,5], stride=[2,2], nonlinearity=tf.nn.leaky_relu, counters=counters):
    cnum = nr_channel
    x, counters = nn.snconv2d(x, cnum, filter_size=[5,5], stride=[2,2], nonlinearity=tf.nn.leaky_relu, counters=counters)
    x, counters = nn.snconv2d(x, cnum*2, filter_size=[5,5], stride=[2,2], nonlinearity=tf.nn.leaky_relu, counters=counters)
    x, counters = nn.snconv2d(x, cnum*4, filter_size=[5,5], stride=[2,2], nonlinearity=tf.nn.leaky_relu, counters=counters)
    x, counters = nn.snconv2d(x, cnum*4, filter_size=[5,5], stride=[2,2], nonlinearity=tf.nn.leaky_relu, counters=counters)
    x, counters = nn.snconv2d(x, cnum*4, filter_size=[5,5], stride=[2,2], nonlinearity=tf.nn.leaky_relu, counters=counters)
    x, counters = nn.snconv2d(x, cnum*4, filter_size=[5,5], stride=[2,2], nonlinearity=tf.nn.leaky_relu, counters=counters)
    x_out = tf.layers.flatten(x)

    return x_out


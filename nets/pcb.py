from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pdb

slim = tf.contrib.slim


def LeakyRelu(x, leak=0.1):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * tf.abs(x)

def pcb_net(inputs,
            end_points,
            num_classes,
            feature_dim=256,
            use_asoftmax=False,
            num_parts=6,
            output_layer="h",
            is_training=True,
            only_pcb=True):
    """
    spilt net (before global avrage pooling) to multi-branch for training
    for resnet_v1_50, set output_stride=4 ensure feature map size
    (tensor to split) before pool5 is 24 * 8

    Args:
        inputs : feature maps before global avrage pooling
        branch_shape : branch"s feature map shape after split
        num_part : numbers of parts to split i.e numbers of branch
        output_layer : features layer
                       "g" : layer after avrage pooling
                       "h" : layer after "g" , before logis layer
    Returns:
        pcb net output
    """
    vector_g = []  # arfter avg pool
    vector_h = []  # after 1*1 conv
    logits = []

    # RPP
    with tf.variable_scope('part_classifier'):
        refined_part = slim.conv2d(inputs, num_parts, [1, 1], 
                                    stride=1, activation_fn=None,
                                    normalizer_fn=None, scope="refined_part")
        refined_part = slim.softmax(refined_part)
        end_points["refined_part"] = refined_part
    if not only_pcb:
        tmp_inputs = tf.expand_dims(inputs, axis=-1)
        refined_part = tf.expand_dims(refined_part, axis=-2)
        tmp_res = tmp_inputs * refined_part
        branches = tf.unstack(tmp_res, axis=-1)
    else:
        branches = tf.split(inputs, num_parts, axis=1)

    with tf.variable_scope('pcb'):
        for i in range(len(branches)):
            # branches[i] = slim.dropout(branches[i], keep_prob=0.5)
            branch = tf.reduce_mean(branches[i], [1, 2], name="pool5",
                                    keep_dims=True)

            net = slim.conv2d(branch, feature_dim,
                                   [1, 1], stride=1,
                                   activation_fn=None,
                                   normalizer_fn=None,
                                   scope="feature_%s" % i)
            
            branch = slim.flatten(branch)
            net = slim.flatten(net)
            vector_h.append(net)
            vector_g.append(branch)
            net = slim.batch_norm(net, activation_fn=None)
            net = LeakyRelu(net)

            if is_training:
                net = slim.dropout(net, keep_prob=0.5)

            logits_part = slim.fully_connected(net,
                                                num_classes,
                                                activation_fn=None,
                                                scope="logits_%s" % i)
            logits.append(logits_part)
            end_points["predictions_%s" % i] = slim.softmax(logits_part,
                                                                scope="predictions")

    vector_h_concat = tf.concat([v for v in vector_h], axis=1)
    vector_g_concat = tf.concat([g for g in vector_g], axis=1)

    end_points["Logits"] = logits
    end_points["h"] = vector_h_concat
    end_points["g"] = vector_g_concat
    net = logits
    return net, end_points

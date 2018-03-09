from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pdb

slim = tf.contrib.slim


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
    if not only_pcb:
        tmp_inputs = tf.expand_dims(inputs, axis=-1)
        refined_part = tf.expand_dims(refined_part, axis=-2)
        tmp_res = tmp_inputs * refined_part
        branches = tf.unstack(tmp_res, axis=-1)
        pdb.set_trace()
    else:
        branches = tf.split(inputs, num_parts, axis=1)

    with tf.variable_scope('pcb'):
        for i in range(len(branches)):
            branch = tf.reduce_mean(branches[i], [1, 2], name="pool5",
                                    keep_dims=True)
            fc5_part = slim.conv2d(branch, int(feature_dim / num_parts),
                                   [1, 1], stride=1,
                                   activation_fn=None,
                                   normalizer_fn=None,
                                   scope="feature_%s" % i)
            net = slim.flatten(fc5_part)
            if num_classes < 5000:
                net = slim.dropout(net, keep_prob=0.8)
            vector_h.append(net)
            vector_g.append(slim.flatten(branch))

            if is_training:
                logits_part = slim.fully_connected(net,
                                                   num_classes,
                                                   activation_fn=None,
                                                   scope="logits_%s" % i)
                logits.append(logits_part)
                end_points["predictions_%s" % i] = slim.softmax(logits_part,
                                                                scope="predictions")
        # pdb.set_trace()

    vector_h_concat = tf.concat([v for v in vector_h], axis=1)
    vector_g_concat = tf.concat([g for g in vector_g], axis=1)
    # if not is_training:
    #     if output_layer == "g":
    #         return vector_g_concat
    #     else:
    #         return vector_h_concat

    end_points["Logits"] = logits
    # end_points["fc5"] = vector_h_concat
    end_points["h"] = vector_h_concat
    end_points["g"] = vector_h_concat
    net = logits
    return net, end_points

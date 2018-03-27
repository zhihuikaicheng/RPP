from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pdb

from nets import resnet_v2
from nets import pcb

FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim

class BaseModel(object):
    def __init__(self):
        '''
        we should declear:
        self.logits
        self.end_points
        self.loss
        self.acc
        '''

class MyResNet(BaseModel):
    def __init__(self, num_classes, sizes, scope, output_stride, spatial_squeeze, reuse, global_pool, is_training=True):
        self.num_classes = num_classes
        self.height = sizes[0]
        self.width = sizes[1]
        self.scope = scope
        self.is_training = is_training
        self.output_stride = output_stride
        self.spatial_squeeze = spatial_squeeze
        self.reuse = reuse
        self.global_pool = global_pool

        with tf.variable_scope(scope):
            self.init_input()
            with slim.arg_scope(resnet_v2.resnet_arg_scope()):
                self.init_network()
            self.init_loss()

    def init_input(self):
        self.image = tf.placeholder(tf.float32, [None, FLAGS.origin_height, FLAGS.origin_width, FLAGS.origin_channel])
        self.label = tf.placeholder(tf.float32, [None, self.num_classes])

    def init_network(self):
        self.sub_models = []

        sub_model = SubResNet([self.image,self.label],
                self.num_classes, self.height, self.width, 'branch_0', is_training=self.is_training,
                output_stride=self.output_stride, spatial_squeeze=self.spatial_squeeze, reuse=self.reuse, global_pool=self.global_pool)
        self.sub_models.append(sub_model)

        self.logits = self.sub_models[0].logits
        self.end_points = self.sub_models[0].end_points
        # self.feature = self.sub_models[0].end_points["global_pool"]

    def init_loss(self):
        cross_entropy = tf.reduce_sum([model.loss for model in self.sub_models])

        regular_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regularizers = tf.add_n(regular_vars)

        self.loss = cross_entropy + FLAGS.weight_decay * regularizers

        tf.summary.scalar('losses/%s_cross_entropy' % self.scope, cross_entropy)
        tf.summary.scalar('losses/%s_regularizers' % self.scope, regularizers)
        tf.summary.scalar('losses/%s' % self.scope, self.loss)

    def load_pretrain_model(self, sess, path):
        # make sure self.scope is the root scope
        for model in self.sub_models:
            model.load_pretrain_model(sess, path, self.scope)

class SubResNet(BaseModel):
    def __init__(self, input, num_classes, height, width, scope, output_stride, spatial_squeeze, reuse, global_pool, is_training=True):
        self.image = input[0]
        self.label = input[1]
        self.num_classes = num_classes
        self.height = height
        self.width = width
        self.scope = scope
        self.is_training = is_training
        self.output_stride = output_stride
        self.global_pool = global_pool
        self.reuse = reuse
        self.spatial_squeeze = spatial_squeeze

        with tf.variable_scope(self.scope):
            self.init_network()
            self.init_loss()

    def init_network(self):
        x = self.image
        x = tf.image.resize_images(x, [self.height, self.width], 0)
        x = tf.subtract(x, 0.5)
        x = tf.multiply(x, 2.0)

        _, end_points = resnet_v2.resnet_v2_50(
            x,
            is_training=self.is_training,
            global_pool=self.global_pool,
            output_stride=self.output_stride,
            spatial_squeeze=self.spatial_squeeze,
            num_classes=None,
            reuse=self.reuse,
            scope='resnet_v2_50'
        )

        with tf.variable_scope('finetune'):
            net = end_points['global_pool']
            net = slim.conv2d(net, 512, [1, 1], stride=1, 
                                activation_fn=None, normalizer_fn=None)
            net = slim.batch_norm(net, activation_fn=None)
            self.feature = net
            net = slim.dropout(net, 0.5)

            net = slim.conv2d(net, self.num_classes, [1, 1], stride=1, 
                            activation_fn=None, normalizer_fn=None)
            net = tf.squeeze(net, [1, 2])
            
        self.logits = net
        self.pred = slim.softmax(net)
        # self.pred = end_points['predictions']
        # self.pred = tf.reduce_mean([end_points['predictions_0'],end_points['predictions_1'],
        #     end_points['predictions_2'],end_points['predictions_3'],
        #     end_points['predictions_4'],end_points['predictions_5']], axis=0)
        self.end_points = end_points

        corr_pred = tf.equal(tf.argmax(self.label,1), tf.argmax(self.pred,1))
        self.acc = tf.reduce_sum(tf.cast(corr_pred, tf.int32))

        for end_point in self.end_points:
            x = self.end_points[end_point]
            tf.summary.histogram('activations/%s/%s'%(self.scope,end_point), x)
            tf.summary.scalar('sparsity/%s/%s'%(self.scope,end_point), tf.nn.zero_fraction(x))

        tf.summary.scalar('acc/%s' % self.acc, self.acc)

    def init_loss(self):
        cross_entropy = -tf.reduce_sum(self.label*tf.log(self.pred + FLAGS.opt_epsilon), axis=1)
        # for i in range(len(self.logits)):
        #     cross_entropy += -tf.reduce_sum(self.label*tf.log(self.end_points["predictions_%s" % i]+FLAGS.opt_epsilon), axis=1)
        self.loss = tf.reduce_mean(cross_entropy)

        tf.summary.scalar('losses/%s' % self.scope, self.loss)

    def load_pretrain_model(self, sess, path, father_scope):
        '''
        in pretrain, name like:InceptionV3/Mixed_7b/Branch_2/Conv2d_0d_3x1/weights
        in our model, name like:inception_v3/branch_0/InceptionV3/Mixed_7b/Branch_2/Conv2d_0d_3x1/weights:0
        so, model_name = inception_v3/branch_0/ + pretrain_name + ':0'
                       = father_scope/sub_model_scope/pretrain_name:0

        note:
        some vars can't be load like final-fc
        in pretrain, name of these vars like:InceptionV3/AuxLogits/Conv2d_2b_1x1/biases
                                             InceptionV3/Logits/Conv2d_2b_1x1/biases
        they all have prefix like InceptionV3/AuxLogits/ or InceptionV3/Logits/
        '''
        scope = father_scope + '/' + self.scope + '/'
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

        d = {}
        for var in variables:
            name = var.name.replace(scope, '').replace(':0', '')
            if name.startswith('resnet_v2_50/logits') or name.startswith('pcb') or name.startswith('part_classifier') or name.startswith('finetune'):
                continue
            d[name] = var

        saver = tf.train.Saver(d)
        saver.restore(sess, path)

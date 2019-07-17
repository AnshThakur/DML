from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import keras

def inference(images, keep_probability, phase_train=True, bottleneck_layer_size=128, weight_decay=0.0, reuse= tf.AUTO_REUSE):
	batch_norm_params = {
		'decay': 0.995,
		'epsilon': 0.001,
		'updates_collections': None,
		'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
	}

	with slim.arg_scope([slim.conv2d, slim.fully_connected],
		weights_initializer=slim.initializers.xavier_initializer(),
		weights_regularizer=slim.l2_regularizer(weight_decay),
		normalizer_fn=slim.batch_norm,
		normalizer_params=batch_norm_params):
		return conv_net(images, is_training=phase_train, dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size, reuse=reuse)




def conv_net(inputs, is_training=True, dropout_keep_prob=0.8, bottleneck_layer_size=128, reuse=None, scope='cnn'):
    
    o1 = keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation="relu")(inputs)
  
    m1 = keras.layers.Conv2D(64, (1, 1), strides=(2, 1), padding="same", activation='relu')(o1)
    m2_1 = keras.layers.Conv2D(64, (1, 1), strides=(1, 1), padding="same", activation='relu')(o1)
    m2_2 = keras.layers.Conv2D(64, (3, 3), strides=(2, 1), padding="same", activation='relu')(m2_1)
    m3_1 = keras.layers.Conv2D(64, (1, 1), strides=(1, 1), padding="same", activation='relu')(o1)
    m3_2 = keras.layers.Conv2D(64, (5, 5), strides=(1, 1), padding="same", activation='relu')(m3_1)
    m3_3 = keras.layers.Conv2D(64, (5, 5), strides=(2, 1), padding="same", activation="relu")(m3_2)
    o1=keras.layers.concatenate([m1,m2_2])
    o1=keras.layers.concatenate([o1,m3_3])

    m1 = keras.layers.Conv2D(64, (1, 1), strides=(2, 1), padding="same", activation='relu')(o1)
    m2_1 = keras.layers.Conv2D(64, (1, 1), strides=(1, 1), padding="same", activation='relu')(o1)
    m2_2 = keras.layers.Conv2D(64, (3, 3), strides=(2, 1), padding="same", activation='relu')(m2_1)
    m3_1 = keras.layers.Conv2D(64, (1, 1), strides=(1, 1), padding="same", activation='relu')(o1)
    m3_2 = keras.layers.Conv2D(64, (5, 5), strides=(1, 1), padding="same", activation='relu')(m3_1)
    m3_3 = keras.layers.Conv2D(64, (5, 5), strides=(2, 1), padding="same", activation="relu")(m3_2)
    o1=keras.layers.concatenate([m1,m2_2])
    o1=keras.layers.concatenate([o1,m3_3])


    m1 = keras.layers.Conv2D(64, (1, 1), strides=(2, 1), padding="same", activation='relu')(o1)
    m2_1 = keras.layers.Conv2D(64, (1, 1), strides=(1, 1), padding="same", activation='relu')(o1)
    m2_2 = keras.layers.Conv2D(64, (3, 3), strides=(2, 1), padding="same", activation='relu')(m2_1)
    m3_1 = keras.layers.Conv2D(64, (1, 1), strides=(1, 1), padding="same", activation='relu')(o1)
    m3_2 = keras.layers.Conv2D(64, (5, 5), strides=(1, 1), padding="same", activation='relu')(m3_1)
    m3_3 = keras.layers.Conv2D(64, (5, 5), strides=(2, 1), padding="same", activation="relu")(m3_2)
    o1=keras.layers.concatenate([m1,m2_2])
    o1=keras.layers.concatenate([o1,m3_3])
    
    m1 = keras.layers.Conv2D(64, (1, 1), strides=(2, 1), padding="same", activation='relu')(o1)
    m2_1 = keras.layers.Conv2D(64, (1, 1), strides=(1, 1), padding="same", activation='relu')(o1)
    m2_2 = keras.layers.Conv2D(64, (3, 3), strides=(2, 1), padding="same", activation='relu')(m2_1)
    m3_1 = keras.layers.Conv2D(64, (1, 1), strides=(1, 1), padding="same", activation='relu')(o1)
    m3_2 = keras.layers.Conv2D(64, (5, 5), strides=(1, 1), padding="same", activation='relu')(m3_1)
    m3_3 = keras.layers.Conv2D(64, (5, 5), strides=(2, 1), padding="same", activation="relu")(m3_2)
    o1=keras.layers.concatenate([m1,m2_2])
    o1=keras.layers.concatenate([o1,m3_3])

    o1 = keras.layers.Conv2D(128, (5, 5), strides=(5, 1), padding="same", activation='relu')(o1)
    o1 = keras.layers.Reshape((501, 128))(o1)
    o1 = keras.layers.Conv1D(128, (501), strides=(1))(o1)
    o1 = keras.layers.Reshape((128,))(o1)

    return o1,o1








def loss_net(inputs, is_training=True, dropout_keep_prob=0.8, bottleneck_layer_size=128, reuse=None, scope='loss'):
    end_points = {}
  
    with tf.variable_scope(scope, 'loss_model', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):

                with tf.variable_scope('Logits'):
                    net = slim.flatten(inputs)
                    net = slim.fully_connected(net, 1024, activation_fn=None, 
                        scope='Bottleneck', reuse=False)          
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='Dropout')
                    net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, 
                        scope='Bottleneck', reuse=False)
          
                    end_points['PreLogitsFlatten'] = net
                
  
    return net, end_points

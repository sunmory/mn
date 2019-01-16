# coding: utf-8

import os
import datetime
import pickle as pkl
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
tf.enable_eager_execution()


params = {
    "filter_length": [3, 4, 5],
    "embedding_size": 5,
    "kernel_num": 2,


}


class CnnModel(tf.keras.Model):
    def __init__(self, params, feature_num, layer_num):
        super().__init__()
        self.params = params
        self.feature_num = feature_num
        self.layer_num = layer_num
        self.build_model()

    def build_model(self):
        self.conv_list, self.pool_list = [], []

        for layer in range(self.layer_num):
            with tf.name_scope("the-{}th-conv-maxpooling".format(str(layer))):
                conv = tf.keras.layers.Conv2D(
                    filters=self.params["kernel_num"],
                    kernel_size=[self.params["filter_length"][0], self.params["embedding_size"]],
                    strides=[1, 1],
                    padding="valid",
                    activation=tf.nn.relu,
                    name="conv",

                )
                pool = tf.keras.layers.MaxPooling2D(pool_size=[self.feature_num - self.params["filter_length"][0] + 1, 1],
                                                          strides=1)
            self.conv_list.append(conv)
            self.pool_list.append(pool)

        with tf.name_scope("full_connect_layer"):
            self.dens1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
            self.dens2 = tf.keras.layers.Dense(units=1)

    def call(self, inputs, training=None, mask=None):
        inputs = tf.reshape(inputs, [-1, self.feature_num, self.params["embedding_size"], 1])
        print(inputs.shape)
        pooling_results = []
        for layer in range(self.layer_num):
            x = self.conv_list[layer](inputs)
            pooling_result = self.pool_list[layer](x)
            pooling_results.append(pooling_result)

        pooling_result = tf.concat(pooling_results, 3)
        flat_x = tf.reshape(pooling_result, (-1, self.params["kernel_num"] * len(self.params["filter_length"])))
        hidden_x = self.dens1(flat_x)
        results = self.dens2(hidden_x)
        return results


def train():
    pass


if __name__ == '__main__':
    train_data = np.arange(500000).reshape((1000, 100, 5))
    print(train_data.shape)
    cnn = CnnModel(params, 100, 3)
    results = cnn(tf.convert_to_tensor(train_data, dtype=tf.float32))
    print(results)
    # print(tf.reshape(results[0], [-1, 4]))
    # print(tf.concat(results, 3))
    # print(cnn.flatten(results[0]))
    print(type(results))

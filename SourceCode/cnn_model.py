# coding: utf-8

import os
import datetime
import pickle as pkl
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool
from multiprocessing import cpu_count
tf.enable_eager_execution()

params = {
    "filter_length": [3, 4, 5],
    "embedding_size": 5,
    "kernel_num": 2,


}


class CnnModel(tf.keras.Model):
    def __init__(self, params, feature_num, layer_num=3):
        super(CnnModel, self).__init__()
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
                    kernel_size=[self.params["filter_length"][layer], self.params["embedding_size"]],
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


def train(feature_data, label_data, train_epoch, batch_size):
    kfolder = KFold(5)
    kfold = kfolder(feature_data, label_data)

    train_index, test_index = next(kfold)

    cnn_model = CnnModel(params, train_epoch.shape[1])
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    for epoch in range(train_epoch):
        batch_index = np.random.randint(0, len(train_index), batch_size)
        train_x = feature_data[batch_index, :]
        train_y = feature_data[batch_index, :]
        with tf.GradientTape() as tape:
            predict_y = cnn_model(tf.convert_to_tensor(train_x))
            loss = tf.losses.mean_squared_error(tf.convert_to_tensor(train_y), predict_y)
        grads = tape.gradient(loss, cnn_model.variables)
        optimizer.apply_gradients(grads_and_vars=(grads, cnn_model.variables))

        if epoch % 10 == 0:
            predict_y = cnn_model(feature_data[test_index, :])
            _loss = tf.losses.mean_squared_error(tf.convert_to_tensor(label_data[test_index]), predict_y)
            print("epoch {} loss {}".format(epoch, _loss))


def cnn_data_generate(data_path):

    def filter_abnormal_data(dataframe):
        """
        过滤异常数据
        :param dataframe:
        :return:
        """
        for column in dataframe.columns:
            if column == "vid": continue
            data_series = dataframe[column][dataframe[column].notnull()]
            low_limit = np.percentile(data_series.values, 0.05)
            height_limit = np.percentile(data_series.values, 0.95)
            data_series[data_series < low_limit] = low_limit
            data_series[data_series > height_limit] = height_limit
            dataframe[column] = data_series
        return dataframe

    def choose_cnn_feature(all_data):
        # all_data = pd.read_csv(data_path + "cnn_all_data.csv", encoding="utf-8", index_col=0)
        sample_num, feature_num = all_data.shape
        feature_not_none_nums = all_data.count()

        for feature_name in feature_not_none_nums.index:
            if feature_not_none_nums.loc[feature_name] / sample_num < 0.01:
                all_data.drop(feature_name, axis=1, inplace=True)

        return all_data

    def fill_none_feature(all_data, type):
        if type == "num":
            mean_data = all_data.mean()

            for feature in all_data.columns:
                all_data[feature][all_data[feature].isnull()] = mean_data.loc[feature]

        if type == "one_hot":
            mode_data = all_data.mode().loc[0]

            for feature in all_data.columns:
                all_data[feature][all_data[feature].isnull()] = mode_data[feature]

        if type == "doc":
            for feature in all_data.columns:
                all_data[feature][all_data[feature].isnull()] = 0

        return all_data

    # num_data

    num_feature_1 = pd.read_csv(data_path + "feature_nums_df.csv", encoding="utf-8", index_col=0)
    num_feature_2 = pd.read_csv(data_path + "feature_nums2_df.csv", encoding="utf-8", index_col=0)
    num_feature_1 = filter_abnormal_data(num_feature_1)
    num_feature_2 = filter_abnormal_data(num_feature_2)
    num_feature = pd.merge(num_feature_1, num_feature_2, how="inner", on=["vid"])
    num_feature = choose_cnn_feature(num_feature)
    num_feature = fill_none_feature(num_feature, "num")

    # num_feature.to_csv(data_path + "cnn_num_feature.csv", encoding="utf-8")

    # one-hot data

    # one_hot_feature = pd.read_csv(data_path + "feature_short_str2.csv", encoding="utf-8", index_col=0)
    # one_hot_feature = choose_cnn_feature(one_hot_feature)
    # one_hot_feature = fill_none_feature(one_hot_feature, "one_hot")
    #
    # one_hot_feature.to_csv(data_path + "cnn_one_hot_feature.csv", encoding="utf-8")

    # dec2vec data

    # doc_2_vec_feature = pd.read_csv(data_path + "feature_d2v.csv", encoding="utf-8", index_col=0)
    # doc_2_vec_feature = choose_cnn_feature(doc_2_vec_feature)
    # doc_2_vec_feature = fill_none_feature(doc_2_vec_feature, "doc")
    # #
    # doc_2_vec_feature.to_csv(data_path + "cnn_doc_2_cev_feature.csv", encoding="utf-8")

    # label
    train_dataframe = num_feature
    for predict_index, predict_item in enumerate(['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']):
        label_dataframe = pd.read_csv(data_path + "throw_nan_label_" + predict_item + ".csv",
                                      encoding="utf-8",
                                      index_col=0,
                                      engine="python")

        train_dataframe = train_dataframe.merge(label_dataframe, how="inner", on="vid")
        label_list = list(train_dataframe[predict_item])
        with open(data_path + str(predict_index) + "_cnn_label.txt", "wb") as f:
            pkl.dump(label_list, f)



def change_data(process_id, num_feature, one_hot_feature, doc_2_vec_feature):
    print("procecss {} start".format(process_id))
    cnn_feature = []

    for sample_id in num_feature.index:
        # print("sample id:{}".format(sample_id))
        sample_feature = []

        # add num feature
        # print("add num feature")
        for feature_name in num_feature.columns:
            sample_feature.append([num_feature.loc[sample_id, feature_name]] * 5)

        # print("add one hot feature")
        for feature_name in one_hot_feature.columns:
            sample_feature.append([one_hot_feature.loc[sample_id, feature_name]] * 5)

        # print("add doc feature")
        for feature_index in range(int(doc_2_vec_feature.shape[1] / 5)):
            f = doc_2_vec_feature.loc[sample_id, doc_2_vec_feature.columns[feature_index * 5: (feature_index + 1) * 5]]
            sample_feature.append(list(f))

        cnn_feature.append(sample_feature)

    # print("process {} finish".format(process_id))
    return cnn_feature


def change_2_cnn_feature_mul_process(data_path):
    # change to CNN feature DataFrame

    num_feature = pd.read_csv(data_path + "cnn_num_feature.csv", encoding="utf-8", index_col=0)
    one_hot_feature = pd.read_csv(data_path + "cnn_one_hot_feature.csv", encoding="utf-8", index_col=0)
    doc_2_vec_feature = pd.read_csv(data_path + "cnn_doc_2_cev_feature.csv", encoding="utf-8", index_col=0)

    procecss_num = cpu_count() - 1
    sample_index = np.linspace(0, num_feature.shape[0], procecss_num + 1, dtype=np.int32)

    cnn_feature = []

    pool = Pool(procecss_num)
    pools = [pool.apply_async(change_data, args=(i + 1,
                                     num_feature.iloc[sample_index[i]: sample_index[i + 1]],
                                     one_hot_feature.iloc[sample_index[i]: sample_index[i + 1]],
                                     doc_2_vec_feature.iloc[sample_index[i]: sample_index[i + 1]]))
                 for i in range(len(sample_index) - 1)]

    pool.close()
    pool.join()

    results = [p.get() for p in pools]

    for result in results:
        cnn_feature.extend(result)

    with open(data_path + "cnn_feature.pkl", "wb") as f:
        pkl.dump(cnn_feature, f)


if __name__ == '__main__':
    # train_data = np.arange(500000).reshape((1000, 100, 5))
    # print(train_data.shape)
    # cnn = CnnModel(params, 100, 3)
    # results = cnn(tf.convert_to_tensor(train_data, dtype=tf.float32))
    # print(results)
    # # print(tf.reshape(results[0], [-1, 4]))
    # # print(tf.concat(results, 3))
    # # print(cnn.flatten(results[0]))
    # print(type(results))

    data_path = "..\\data\\"
    cnn_data_generate(data_path)
    # change_2_cnn_feature_mul_process(data_path)
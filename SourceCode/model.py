# encoding: utf-8

import os
import datetime
import pandas as pd
import numpy as np
import lightgbm as lgbm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


class LstmModel():
    def __init__(self, params, data_path, model_path):
        self.params = params
        self.data_path = data_path
        self.model_path = model_path
        self.epoch = 8
        self.train_dict = {
                'feature_decay': [5, 35, 5, 5, 5, 5, 5],
                'last_lost': [1],
                'is_start': True,
                'lr': [0.018, 0.018, 0.03, 0.04, 0.04]
            }
        now = datetime.datetime.now()
        self.time_stamp = now.strftime("%Y-%m-%d %H-%M-%S_")

    def load_data(self):
        """
        读取数据
        :return:
        """
        # num_data

        num_feature_1 = pd.read_csv(self.data_path + "feature_nums_df.csv", encoding="utf-8", index_col=0)
        num_feature_2 = pd.read_csv(self.data_path + "feature_nums2_df.csv", encoding="utf-8", index_col=0)

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

        num_feature_1 = filter_abnormal_data(num_feature_1)
        num_feature_2 = filter_abnormal_data(num_feature_2)
        train_dataframe = pd.merge(num_feature_1, num_feature_2, how="inner", on=["vid"])
        print(train_dataframe.shape)

        # one-hot data

        one_hot_feature = pd.read_csv(self.data_path + "feature_short_str2.csv", encoding="utf-8", index_col=0)
        train_dataframe = train_dataframe.merge(one_hot_feature, how="inner", on=["vid"])

        # dec2vec data

        doc_2_vec_feature = pd.read_csv(self.data_path + "feature_d2v.csv", encoding="utf-8", index_col=0)
        train_dataframe = train_dataframe.merge(doc_2_vec_feature, how="inner", on=["vid"])

        self.train_dataframe = train_dataframe

    def train_for_each_predict_items(self, predict_index, predict_item, train_epochs):
        model_list = []
        # label data
        label = pd.read_csv(self.data_path + "throw_nan_label_" + predict_item + ".csv",
                            encoding="utf-8",
                            index_col=0,
                            engine="python")
        train_dataframe = self.train_dataframe.merge(label, how="inner", on=["vid"])

        params = self.params
        params["learning_rate"] = self.train_dict["lr"][predict_index]

        def find_one_hot_feature(feature):
            category_feature_path = self.data_path + "category_feature_map.npy"
            all_category_feature = np.load(category_feature_path)
            category_feature = (set(all_category_feature) & set(feature)) - set(["vid"])
            return list(category_feature)

        if os.path.exists(self.important_feature_path):
            feature = pd.read_csv(self.important_feature_path, encoding="utf-8", index_col=0)
            split_frequency = np.percentile(feature["use_num"].values, self.train_dict["feature_decay"][train_epochs])
            feature = feature["check_items"][feature["use_num"] > split_frequency].tolist()
        else:
            feature = train_dataframe.columns
            feature = feature.drop([predict_item])

        categorical_feature = find_one_hot_feature(feature)

        kfold = KFold(5, shuffle=True)
        train_generator = kfold.split(train_dataframe)
        loss = 0
        epoch_num = 1

        for i in range(epoch_num):
            train_index, valid_index = next(train_generator)
            print("train {} model, {} epoch".format(predict_item, i))
            _train_data, _valid_data = train_dataframe.iloc[train_index], train_dataframe.iloc[valid_index]
            train_dataset= lgbm.Dataset(_train_data[feature], _train_data[predict_item])
            valid_dataset = lgbm.Dataset(_valid_data[feature], _valid_data[predict_item])
            model = lgbm.train(params, train_dataset, num_boost_round=3000,
                               valid_sets=valid_dataset, verbose_eval=100, feval=None,
                               early_stopping_rounds=100,  categorical_feature=categorical_feature)
            model_list.append(model)
            valid_predicted = model.predict(_valid_data[feature])
            loss = loss + mean_squared_error(_valid_data[predict_item], valid_predicted)

        # important_feature = pd.Series(model.feature_importance(), index=feature).sort_values(ascending=False)
        important_feature = pd.DataFrame({"check_items": model.feature_name(), "use_num": model.feature_importance()})

        return model_list, important_feature, loss/epoch_num

    def train(self, train_epoch_num):
        self.load_data()
        predict_items_list = ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']

        for predict_index, predict_item in enumerate(predict_items_list):
            # 按每个指标选择不同的最优特征
            self.important_feature_path = self.data_path + self.time_stamp + str(predict_index) + "_import_feature.csv"
            for epoch in range(train_epoch_num):
                model_list, important_feature, loss = self.train_for_each_predict_items(predict_index, predict_item, epoch - 1)
                # important_feature = pd.DataFrame(important_feature, columns=["check_items", "use_num"]).T

                if not os.path.exists(self.important_feature_path):
                    all_feature_loss, last_loss = loss, loss
                    important_feature.to_csv(self.important_feature_path, encoding="utf-8")
                    important_feature.to_csv(self.data_path + self.time_stamp + " all_feature.csv", encoding="utf-8")
                    self.save_model(predict_item, model_list)
                    print("第一次训练，全特征损失：{}".format(all_feature_loss))
                    continue

                if loss < last_loss:
                    print("预测项：{}，轮次：{}，得到缩减特征后的更优模型。".format(predict_item, epoch))
                    print("此前最优模型损失：{}，当前最优模型损失：{}".format(last_loss, loss))
                    important_feature.to_csv(self.important_feature_path, encoding="utf-8")
                    self.save_model(predict_item, model_list)
                    last_loss = loss
                else:
                    print("预测项：{}，轮次：{}，未出现更优模型。".format(predict_item, epoch))

    def save_model(self, predict_item, model_list):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        for model_index, model in enumerate(model_list):
            model.save_model(self.model_path + predict_item + "_model_" + str(model_index) + ".txt")


if __name__ == '__main__':
    params = {
        'learning_rate': 0.025,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mse',  # 使用均方误差
        'num_leaves': 60,  # 最大叶子数for base learner
        'feature_fraction': 0.6,  # 选择部分的特征
        'min_data': 100,  # 一个叶子上的最少样本数
        'min_hessian': 1,  # 一个叶子上的最小 hessian 和，子叶权值需大于的最小和
        'verbose': 1,
        'lambda_l1': 0.3,  # L1正则项系数
        'device': 'cpu',
        'num_threads': 8,  # 最好设置为真实核心数
    }
    data_path = "..\\data\\"
    model_path = "..\\model\\"
    lstm_model = LstmModel(params, data_path, model_path)
    lstm_model.train(8)

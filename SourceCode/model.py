# encoding: utf-8

import os
import datetime
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


class LstmModel():
    def __init__(self, params, data_path, result_path):
        self.params = params
        self.data_path = data_path
        self.result_path = result_path

    def load_data(self):
        """
        读取数据
        :return:
        """
        # num_data

        num_feature_1 = pd.read_csv(self.data_path + "feature_nums_df.csv", encoding="utf-8", index_col=0)
        num_feature_2 = pd.read_csv(self.data_path + "feature_nums2_df.csv", encoding="utf-8", index_col=0)

        def filter_abnormal_data(dataframe):
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
        feature = pd.merge(num_feature_1, num_feature_2, how="inner", on=["vid"])
        print(feature.shape)

        # one-hot data

        one_hot_feature = pd.read_csv(self.data_path + "feature_short_str2.csv", encoding="utf-8", index_col=0)
        feature = feature.merge(one_hot_feature, how="inner", on=["vid"])

        doc_2_vec_feature = pd.read_csv(self.path + "feature_d2v.csv", encoding="utf-8", index_col=0)
        feature = feature.merge(doc_2_vec_feature, how="inner", on=["vid"])



if __name__ == '__main__':
    params = []
    data_path = "..\\data\\"
    target_path = "..\\data\\"
    lstm_model = LstmModel(params, data_path, target_path)
    lstm_model.load_data()
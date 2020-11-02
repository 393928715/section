import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import tushare as ts
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from zgtools.data_tool import *
import joblib
import os

def section_close_scatter(df_cy50_mkt, close_col):
    # 画各分域收盘价散点图
    df_close_section = pd.DataFrame(columns=range(4), index=df_cy50_mkt.dropna().index)
    for idx in df_cy50_mkt.dropna().index:
        section = df_cy50_mkt.loc[idx, 'bin']
        df_close_section.loc[idx, section] = df_cy50_mkt.loc[idx, close_col]
    mask_size = 5
    fig = plt.figure(figsize=(9, 5))
    plt.scatter(df_close_section.index, df_close_section.iloc[:, 0], mask_size)
    plt.legend('0')
    plt.scatter(df_close_section.index, df_close_section.iloc[:, 1], mask_size)
    plt.legend('1')
    plt.scatter(df_close_section.index, df_close_section.iloc[:, 2], mask_size)
    plt.legend('2')
    plt.scatter(df_close_section.index, df_close_section.iloc[:, 3], mask_size)
    plt.legend([0, 1, 2, 3])
    plt.show()

def group_box_plot(df, show=False):
    '''
    本函数实现对因子收益在不同分域的统计
    :param df: 因子收益率矩阵
    :return:  返回不同分域的因子收益率箱型图
    '''
    group = df.groupby(level=0)
    boxes = []
    for i, g in group:
        boxes.append(g.values)
    plt.boxplot(boxes, vert=1)
    plt.title(g.columns[0])
    if show:
        plt.show()
    df_mean = group.mean()[g.columns[0]]
    return df_mean

def mean_std_stat(df_cy50_mkt):

    mean1 = df_cy50_mkt[df_cy50_mkt.bin == 0]['next_open2open'].mean()
    mean2 = df_cy50_mkt[df_cy50_mkt.bin == 1]['next_open2open'].mean()
    mean3 = df_cy50_mkt[df_cy50_mkt.bin == 2]['next_open2open'].mean()
    mean4 = df_cy50_mkt[df_cy50_mkt.bin == 3]['next_open2open'].mean()

    std1 = df_cy50_mkt[df_cy50_mkt.bin == 0]['next_open2open'].std()
    std2 = df_cy50_mkt[df_cy50_mkt.bin == 1]['next_open2open'].std()
    std3 = df_cy50_mkt[df_cy50_mkt.bin == 2]['next_open2open'].std()
    std4 = df_cy50_mkt[df_cy50_mkt.bin == 3]['next_open2open'].std()

    df_stat = pd.DataFrame({'mean':[mean1,mean2,mean3,mean4], 'std':[std1,std2,std3,std4]})
    df_stat['mean'].plot(kind='bar', title='mean');plt.show()
    df_stat['std'].plot(kind='bar', title='std');plt.show()

    return df_stat

class ts_clustering:

    def __init__(self, df_factor,model_save_name):
        self.df_factor = df_factor
        self.model_save_path = os.path.join('model', model_save_name)

    def split_data(self, df, date):
        '''
        :param df:索引为datetime的DataFrame
        :param date: 分割数据及的日期，Str格式
        :return:
        '''
        df_train = df[df.index <= date]
        df_test = df[~df.index.isin(df_train.index)]
        return df_train, df_test

    def scale_data(self, df_train, df_test):
        df_train, scaler = data_scale(df_train, df_train.columns)
        df_test, _ = data_scale(df_test, df_test.columns, scaler)
        return df_train, df_test

    def get_model(self, df_train):
        clf = KMeans(n_clusters=4, random_state=0)
        clf.fit(df_train)
        return clf

    def save_model(self, model):
        joblib.dump(model, self.model_save_path)

    def load_model(self, model_path):
        model = joblib.load(model_path)
        return model

    def run(self):
        # 分割数据集
        df_train, df_test = class_cluster.split_data(self.df_factor, date='2020-07-01')
        # 标准化
        df_train, df_test = class_cluster.scale_data(df_train, df_test)
        # 获取模型
        model = class_cluster.get_model(df_train)
        # 预测
        train_bin = model.predict(df_train)
        test_bin = model.predict(df_test)
        # 把聚类标签放入原始数据中
        df_train['bin'] = train_bin
        df_test['bin'] = test_bin
        return df_train, df_test


if __name__ == '__main__':
    # 读取行情数据
    df_cy50_mkt = pd.read_csv('data/159949.txt', encoding='gbk', header=1, sep=r'\t').iloc[:-1]
    df_cy50_mkt['时间'] = pd.to_datetime(df_cy50_mkt['时间'])
    df_cy50_mkt.set_index('时间', inplace=True)

    # 读取分域特征文件
    df_voltility_score = pd.read_csv('data/cyb50_voltility_score_outofsample.csv', index_col=0, names=['voltility_score'],header=1)
    df_ma_score = pd.read_csv('data/cyb50_ma_score_outofsample.csv', index_col=0, names=['ma_score'],header=1)
    df_volume_score = pd.read_csv('data/cyb50_volume_score_outofsample.csv', index_col=0, names=['volume_score'],header=1)
    df = pd.concat([df_ma_score, df_voltility_score, df_volume_score], axis=1)
    df.index = pd.to_datetime(df.index.astype(str))
    df.dropna(axis=0,inplace=True)

    # 初始化
    class_cluster = ts_clustering(df, model_save_name='test.pkl')
    df_train_bin, df_test_bin = class_cluster.run()
    # df_train.to_csv('data/k-means_train_sz50.csv')
    # df_test.to_csv('data/k-means_test_cy50.csv')

    # 统计聚类效果
    df_cy50_mkt['bin'] = df_train_bin['bin']

    # 画出各分域close的散点图
    section_close_scatter(df_cy50_mkt, close_col='    收盘')

    # 统计样本内各分域日收益表现
    df_cy50_mkt['next_open2open'] = df_cy50_mkt.iloc[:,3].pct_change().shift(-1)
    #df_cy50_mkt['next_open2open'] = df_cy50_mkt.iloc[:,0].pct_change().shift(-2)
    df_train_stat = mean_std_stat(df_cy50_mkt)

    # 统计样本外日收益表现
    df_ret = df_cy50_mkt[df_cy50_mkt.index.isin(df_test_bin.index)]
    df_ret['bin'] = df_test_bin['bin']
    df_stat = mean_std_stat(df_ret)
    df_mean = pd.DataFrame()
    df_mean['out-of-sample'] = df_stat['mean']
    df_mean['intro-sample'] = df_train_stat['mean']
    df_mean.plot(kind='bar');plt.show()

    # 回测
    df_ret['order'] = df_ret['bin'].apply(lambda x: 1 if x in [1] else -1 if x in [3] else 0)
    df_pnl = ts_backtest(df_ret, order_col='order', fee_rate=0.002, ret_col='next_open2open')

    # # 读取中证500因子收益率文件
    # df_fac_ret = pd.read_csv('data/fac_ret_000905pnl.csv', index_col=0, parse_dates=True)
    # # 统计各分域收益率表现
    # df_fac_ret['bin'] = df_train_bin['bin']
    # df_fac_ret.dropna(axis=0, inplace=True)
    # df_fac_ret_mean = df_fac_ret.groupby('bin').mean()
    #
    # df_fac_weight = df_fac_ret_mean.copy()
    # # 获得各分域因子权重
    # for i in range(4):
    #     df_fac_weight.iloc[i] = df_fac_ret_mean.iloc[i]/abs(df_fac_ret_mean.iloc[i]).sum()
    #
    # df_fac_weight = df_fac_ret_mean.div(np.abs(df_fac_ret_mean).sum(axis=1), axis='columns')
    #
    # # 读取因子收益率文件
    # df_ret = pd.read_csv(r'data/df_factor_day.csv', index_col=['date', 'code'], usecols=['date','code','next_open2open'])
    # #df_ret = df_ret.unstack()['next_open2open']
    # #df_ret['bin'] = df_train_bin['bin']
    # #df_ret.set_index('bin', inplace=True)
    #
    # columns = ['fac_' + str(i) for i in range(df_ret.shape[1])]
    # df_ret.columns = columns
    #
    # # 统计各因子收益率在不通分域的表现，并绘制箱型图
    # df_mean = df_ret.dropna(axis=0).set_index('bin').groupby(level=0, axis=1).apply(lambda x: group_box_plot(x))
    #
    # df_ic_stat = pd.DataFrame({'0':df_mean.iloc[0], '1':df_mean.iloc[1]})


import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import tushare as ts
from collections import Counter
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
import numpy as np
from zgtools.data_tool import *

def kmeans_cluster(df_train, n_clusters):
    '''
    :param df: 聚类的特征矩阵
    :param n_clusters: 指定分类个数
    :return: 返回聚类预测值
    '''
    clf = KMeans(n_clusters=n_clusters, random_state=0)
    clf.fit(df)
    predict = clf.predict(df)
    return predict

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

def ts_backtest(df_ret, order_col='order', ret_col='next_open2open',
                fee_rate=0.002, title='', show=True, model=0):
    '''
    :param df_ret: 包含下期收益率和交易方向的DataFrame
    :param order_col: 交易方向列名
    :param ret_col: 收益列名
    :param fee_rate: 交易费用比率
    :param show: 是否显示回测曲线
    :param model: 0：单利，1：复利
    :return: 附带pnl曲线的DataFrame
    '''
    df_ret['fee'] = np.abs(df_ret[order_col].diff() * fee_rate)
    df_ret['ret'] = df_ret[order_col] * df_ret['next_open2open'] - df_ret['fee']
    if model==0 :df_ret['pnl'] = df_ret['ret'].cumsum()
    else: df_ret['pnl'] = df_ret['ret'].cumprod()
    if show:df_ret['pnl'].plot(title=title);plt.show()
    return df_ret

def section_trend_scatter(df_cy50_mkt):
    # 画各分域收盘价散点图
    df_close_section = pd.DataFrame(columns=range(4), index=df_cy50_mkt.dropna().index)
    for idx in df_cy50_mkt.dropna().index:
        section = df_cy50_mkt.loc[idx, 'bin']
        df_close_section.loc[idx, section] = df_cy50_mkt.loc[idx, '    收盘']

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

# 读取行情数据
df_cy50_mkt = pd.read_csv('data/159949.txt', encoding='gbk', header=1, sep=r'\t').iloc[:-1]
df_cy50_mkt['时间'] = pd.to_datetime(df_cy50_mkt['时间'])
df_cy50_mkt.set_index('时间', inplace=True)

# 读取分域特征文件
df_voltility_score = pd.read_csv('data/cyb50_voltility_score.csv', index_col=0, names=['voltility_score'],header=1)
df_ma_score = pd.read_csv('data/cyb50_ma_score.csv', index_col=0, names=['ma_score'],header=1)
df_volume_score = pd.read_csv('data/cyb50_volume_score.csv', index_col=0, names=['volume_score'],header=1)
df = pd.concat([df_ma_score, df_voltility_score, df_volume_score], axis=1)
df.index = pd.to_datetime(df.index.astype(str))
df.dropna(axis=0,inplace=True)

df_train = df[df.index <= '2020-01-01']
df_test = df[~df.index.isin(df_train.index)]

# 标准化
df_train, scaler = data_scale(df_train, df_train.columns)
df_test, _ = data_scale(df_test, df_test.columns)

# 聚类
clf = KMeans(n_clusters=4, random_state=0)
clf.fit(df_train)

df_train['bin'] = clf.predict(df_train)
df_train['bin'].value_counts().plot(kind='bar');plt.show()

# 统计聚类效果
df_cy50_mkt['bin'] = df_train['bin']

# 画各分域收盘价散点图
section_trend_scatter(df_cy50_mkt)

# 统计样本内各分域日收益表现
df_cy50_mkt['next_open2open'] = df_cy50_mkt.iloc[:,0].pct_change().shift(-2)
# df_cy50_mkt['next_open2open'] = df_cy50_mkt.iloc[:,3].pct_change().shift(-1)
df_train_stat = mean_std_stat(df_cy50_mkt)

# 做样本内回测
df_ret_train = df_cy50_mkt.dropna(axis=0)
df_ret_train['order'] = df_ret_train['bin'].apply(lambda x: 1 if x in [1] else -1 if x in [2] else 0)
df_ret_train = ts_backtest(df_ret_train, title='样本内净值')

# 预测样本外分类
df_test['bin'] = clf.predict(df_test)
df_ret = df_cy50_mkt[df_cy50_mkt.index.isin(df_test.index)]
df_ret['bin'] = df_test['bin']

# 统计样本外表现
df_stat = mean_std_stat(df_ret)

# 样本外回测
df_ret['order'] = df_ret['bin'].apply(lambda x: 1 if x in [1] else -1 if x in [3] else 0)
df_ret = ts_backtest(df_ret, title='样本外净值')

df_mean = pd.DataFrame()
df_mean['样本内日收益均值'] = df_train_stat['mean']
df_mean['样本外日收益均值'] = df_stat['mean']
df_mean.plot(kind='bar');plt.show()

# 读取中证500因子收益率文件
df_fac_ret = pd.read_csv('data/fac_ret_000905pnl.csv', index_col=0, parse_dates=True)
# 统计各分域收益率表现
df_fac_ret.loc[df_fac_ret.index.isin(df_train.index),'bin'] = df_train['bin']
df_fac_ret.loc[df_fac_ret.index.isin(df_test.index),'bin'] = df_test['bin']
df_fac_ret.dropna(axis=0, inplace=True)
df_fac_ret_mean = df_fac_ret.groupby('bin').mean()

# 获得各分域因子权重
df_fac_weight = df_fac_ret_mean.div(abs(df_fac_ret_mean.sum(axis=1)), axis='rows')

# 获得因子等权收益
df_fac_ret['equal_weight'] = df_fac_ret[df_fac_ret.columns.difference(['bin'])].mean(axis=1)

# 按不同分域分配权重
df_list = []
factor_ret_cols = df_fac_weight.columns.difference(['bin','equal_weight'])
for i in range(4):
    df_fac_ret_tmp = df_fac_ret[df_fac_ret.bin==i]
    df_fac_weight_tmp = df_fac_weight[df_fac_weight.index==i][factor_ret_cols]
    df_factor_ret_tmp = np.multiply(df_fac_ret_tmp[factor_ret_cols], df_fac_weight_tmp.values).sum(axis=1)
    df_list.append(df_factor_ret_tmp)
df_factor_ret_weight = pd.concat(df_list, axis=0)
df_fac_ret['section_weight'] = df_factor_ret_weight

# 样本外加权测试
df_fac_ret.loc[df_fac_ret[df_fac_ret.index>'2020-01-01'].index,['section_weight', 'equal_weight']].cumsum().plot();plt.show()

df_fac_ret_stat = pd.DataFrame({'0':df_fac_ret_mean.iloc[0], '1':df_fac_ret_mean.iloc[1],
                                '2':df_fac_ret_mean.iloc[2], '3':df_fac_ret_mean.iloc[3]})
df_fac_ret_stat.plot(kind='bar');plt.show()

import seaborn as sns
sns.heatmap(df_fac_ret[factor_ret_cols].corr());plt.show()

#df_mean = df_ic.dropna(axis=0).set_index('bin').groupby(level=0, axis=1).apply(lambda x: group_box_plot(x))

#df_fac_ret_tmp[factor_ret_cols].mul(df_fac_weight_tmp.T, axis='columns')
# df_std = pd.DataFrame()

# df_train.to_csv('k-means_train.csv')
# df_test.to_csv('k-means_test.csv')

# df_ic = pd.read_csv('ic.csv', index_col=0, parse_dates=True)
# df_ic['bin'] = df_train['bin']

# # 读取因子收益率文件
# # df_ret = pd.read_csv(r'df_factor_day.csv', index_col=['date', 'code'], usecols=['date','code','next_open2open'])
# # df_ret = df_ret.unstack()['next_open2open']
# # df_ret['bin'] = df['bin']
# # df_ret.set_index('bin', inplace=True)
# #
# # columns = ['fac_' + str(i) for i in range(df_ret.shape[1])]
# # df_ret.columns = columns
#
# # # 统计各因子收益率在不通分域的表现，并绘制箱型图
# df_mean = df_ic.dropna(axis=0).set_index('bin').groupby(level=0, axis=1).apply(lambda x: group_box_plot(x))
#
# df_ic_stat = pd.DataFrame({'0':df_mean.iloc[0], '1':df_mean.iloc[1]})
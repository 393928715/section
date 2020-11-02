import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import tushare as ts
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from zgtools.data_tool import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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

# 读取barra因子文件
df_fac_ret = pd.read_csv('data/barra_fac_ret.csv', index_col=0, parse_dates=True)

# 取16年之后的数据
df_fac_ret = df_fac_ret[df_fac_ret.index >= '2016-01-01']

df_fac_ret = df_fac_ret.shift(1) # 把明日因子收益率前推一天，变成当日的
df_fac_ret.dropna(axis=0,inplace=True)

# 划分训练集测试集
split_date = '2020-01-01'
df_train = df_fac_ret[df_fac_ret.index <= split_date]
df_test = df_fac_ret[df_fac_ret.index > split_date]

# df_train = df_fac_ret.iloc[:int(len(df_fac_ret) * 0.8)]
# df_test = df_fac_ret.iloc[int(len(df_fac_ret) * 0.8):]

# 聚类
clf = KMeans(n_clusters=3, random_state=0)
clf.fit(df_train)

# 得到聚类标签
df_train['bin'] = clf.predict(df_train)
df_test['bin'] = clf.predict(df_test)
df_fac_ret.loc[df_fac_ret.index.isin(df_train.index), 'bin'] = df_train['bin']
df_fac_ret.loc[df_fac_ret.index.isin(df_test.index), 'bin'] = df_test['bin']

# 统计聚类内各因子表现
df_fac_ret_mean = df_train.dropna(axis=0).set_index('bin').groupby(level=0, axis=1).apply(lambda x: group_box_plot(x))

# 获得各分域因子权重
df_fac_weight = df_fac_ret_mean.div(abs(df_fac_ret_mean).sum(axis=1), axis='rows')
#df_fac_weight = abs((df_fac_ret_mean.T/abs(df_fac_ret_mean).sum(axis=1)).T)

# 计算收益前要先将当日收益变成明日收益，用当日标签对明日因子收益做权重分配
df_fac_ret[df_fac_ret.columns.difference(['bin'])] = df_fac_ret[df_fac_ret.columns.difference(['bin'])].shift(-1)

# 获得因子等权收益
df_fac_ret['equal_weight'] = df_fac_ret[df_fac_ret.columns.difference(['bin'])].mean(axis=1)

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

# 按不同分域分配权重
# df_list = []
# factor_ret_cols = df_fac_weight.columns.difference(['bin','equal_weight'])
# for i in df_fac_ret.bin.unique():
#     df_fac_ret_tmp = df_fac_ret[df_fac_ret.bin==i]
#     df_fac_weight_tmp = df_fac_weight[df_fac_weight.index==i][factor_ret_cols]
#     if i == 1:
#         #df_fac_weight_tmp
#         df_fac_weight_tmp.loc[:,:]= 1/df_fac_weight_tmp.shape[1]
#     df_factor_ret_tmp = np.multiply(df_fac_ret_tmp[factor_ret_cols], df_fac_weight_tmp.values).sum(axis=1)
#     df_list.append(df_factor_ret_tmp)
# df_factor_ret_weight = pd.concat(df_list, axis=0)
# df_fac_ret['section_weight_no1'] = df_factor_ret_weight

# df_list = []
# factor_ret_cols = df_fac_weight.columns.difference(['bin','equal_weight'])
# for i in df_fac_ret.bin.unique():
#     df_fac_ret_tmp = df_fac_ret[df_fac_ret.bin==i]
#     df_fac_weight_tmp = df_fac_weight[df_fac_weight.index==i][factor_ret_cols]
#     df_factor_ret_tmp = np.multiply(df_fac_ret_tmp[factor_ret_cols], df_fac_weight_tmp.values).sum(axis=1)
#     df_list.append(df_factor_ret_tmp)
# df_factor_ret_weight = pd.concat(df_list, axis=0)
# df_fac_ret['section_weight'] = df_factor_ret_weight

# 把多空收益区分开
#df_list = []
long_list = []
short_list = []

factor_ret_cols = df_fac_weight.columns.difference(['bin','equal_weight'])
# for i in range(4):
#     df_fac_ret_tmp = df_fac_ret[df_fac_ret.bin==i]
#     df_fac_weight_tmp = df_fac_weight[df_fac_weight.index==i][factor_ret_cols]
#
#     long_weight_cols = df_fac_weight_tmp[df_fac_weight_tmp>0].dropna(axis=1).columns
#     long_weight_tmp = df_fac_weight_tmp[long_weight_cols] / df_fac_weight_tmp[long_weight_cols].sum().sum()
#
#     short_weight_cols = df_fac_weight_tmp[df_fac_weight_tmp<=0].dropna(axis=1).columns
#     short_weight_tmp = df_fac_weight_tmp[short_weight_cols] / abs(df_fac_weight_tmp[short_weight_cols]).sum().sum()
#
#     long_ret_tmp = np.multiply(df_fac_ret_tmp[long_weight_cols], df_fac_weight_tmp[long_weight_cols].values).sum(axis=1)
#     short_ret_tmp = np.multiply(df_fac_ret_tmp[short_weight_cols], df_fac_weight_tmp[short_weight_cols].values).sum(axis=1)
#
#     long_list.append(long_ret_tmp)
#     short_list.append(short_ret_tmp)
#
#     #df_factor_ret_tmp = np.multiply(df_fac_ret_tmp[factor_ret_cols], df_fac_weight_tmp.values).sum(axis=1)
#
#     #df_list.append(df_factor_ret_tmp)
#
# df_long_ret = pd.concat(long_list, axis=0) # 只做多头
# df_short_ret = pd.concat(short_list, axis=0) # 只做空头
# df_factor_ret_weight = pd.concat(df_list, axis=0) # 总收益

# df_fac_ret['section_weight'] = df_factor_ret_weight
# df_fac_ret['long_ret'] = df_long_ret
# df_fac_ret['short_ret'] = df_short_ret

# 样本外加权测试
# df_fac_ret.loc[df_fac_ret[df_fac_ret.index>'2020-01-01'].index,['section_weight', 'equal_weight', 'long_ret', 'short_ret']].cumsum().plot();plt.show()
#
# df_fac_ret.loc[df_fac_ret[df_fac_ret.index>'2020-01-01'].index,['long_ret', 'short_ret']].cumsum().plot();plt.show()
#
# df_fac_ret.loc[df_fac_ret[df_fac_ret.index>'2020-01-01'].index,['section_weight', 'section_weight_no1', 'equal_weight']].cumsum().plot();plt.show()
#
# df_fac_ret_stat = pd.DataFrame({'0':df_fac_ret_mean.iloc[0], '1':df_fac_ret_mean.iloc[1],
#                                 '2':df_fac_ret_mean.iloc[2], '3':df_fac_ret_mean.iloc[3]})
# df_fac_ret_stat.plot(kind='bar', title='train');plt.show()

# 统计各分域因子表现
df_train.groupby('bin').mean().T.sort_index().plot(kind='bar', title='train', figsize=(10,6));plt.show()
df_test.groupby('bin').mean().T.sort_index().plot(kind='bar', title='test', figsize=(10,6));plt.show()

# 读入每日股票池
df_stk_pool = pd.read_csv('data/stk_pool.csv', header=1)
df_stk_pool = df_stk_pool[list(filter(lambda x: x >= '20200101', df_stk_pool.columns))]

# 读入每日股票收益
df_stk_mkt = pd.read_csv('data/stk_mkt.csv', parse_dates=['date'])
df_stk_mkt.code = df_stk_mkt.code.apply(lambda x:int(x[:6]))
df_next_ret = df_stk_mkt[['date', 'code', 'next_open2open']].pivot_table(columns='code', index='date', values='next_open2open')

# # 获得等权买入矩阵
# stk_list = []
# for date in df_stk_pool.columns:
#       stk_list += df_stk_pool.loc[:,date].dropna().tolist()
#
# # stk_list = set(list((df_stk_pool.values.reshape(-1,1))))
# stk_list = list(set(stk_list))
# df_order = pd.DataFrame(index=df_stk_pool.columns, columns=stk_list)
# for date in df_order.index:
#     df_order.loc[date, df_stk_pool.loc[:,date].dropna().tolist()]=1
# df_order.fillna(0, inplace=True)
# df_weight = df_order.groupby(axis=0, level=0).apply(lambda x: x / (np.abs(x).sum().sum()))
#
# df_pnl_equal_weight = backtest(df_weight, df_next_ret[df_weight.columns])

# 读入barra的rank_score
df_barra_rank_score = pd.read_csv('data/barra_rank_score.csv')
df_barra_rank_score.columns = ['date', 'code'] + df_barra_rank_score.columns[2:].tolist()
df_barra_rank_score = df_barra_rank_score[df_barra_rank_score.date >= 20200101]

# 统计每日股票池在barra上的暴露，进行权重分配
# stk_filter_dict = {}
# stk_filter_list = []
df_weight_del_neg_list = []
df_weight_del_neg_rank_list = []

scaler = MinMaxScaler((0.3,1))
for date in df_stk_pool.columns:
    if date in df_test.index:
        seri_stk_day = df_stk_pool[date].dropna()
        df_rank_score_tmp = df_barra_rank_score[(df_barra_rank_score.date.isin([date]))&(df_barra_rank_score.code.isin(seri_stk_day))]
        section = df_test[df_test.index == pd.to_datetime(date)].bin[0]
        df_weight_tmp = df_fac_weight.loc[section]
        
        df_rank_score_tmp['rank'] = np.multiply(df_rank_score_tmp[df_rank_score_tmp.columns.difference(['date','code'])], df_weight_tmp.values).sum(axis=1)
        df_rank_score_tmp.sort_values('rank', ascending=False, inplace=True)
        equal_weight =  [1/len(df_rank_score_tmp) for i in range(len(df_rank_score_tmp))]
        df_rank_score_tmp['rank'] = scaler.fit_transform(df_rank_score_tmp['rank'].values.reshape(-1, 1))
        rank_score_weight = df_rank_score_tmp['rank'] / df_rank_score_tmp['rank'].sum()
        df_rank_score_tmp['weight'] = rank_score_weight
        # if len(df_rank_score_tmp[df_rank_score_tmp['rank'] > 0]) !=0:
            
        #     df_rank_score_tmp['neg']=1
        #     df_rank_score_tmp = df_rank_score_tmp[df_rank_score_tmp['rank'] > 0]
            
        #     equal_weight =  [1/len(df_rank_score_tmp) for i in range(len(df_rank_score_tmp))]
        #     rank_score_weight = df_rank_score_tmp['rank'] / df_rank_score_tmp['rank'].sum()
            
            
        #     df_tmp = df_rank_score_tmp.copy()
        #     df_tmp['weight'] = equal_weight
        #     df_weight_del_neg_list.append(df_tmp)
            
        #     df_rank_score_tmp['weight'] = rank_score_weight
        #     df_weight_del_neg_rank_list.append(df_rank_score_tmp)
        # else:
            
        #     df_rank_score_tmp['weight'] = 1/len(df_rank_score_tmp) # 更好的是应该做风格中性
        #     df_weight_del_neg_list.append(df_rank_score_tmp)
        df_weight_del_neg_rank_list.append(df_rank_score_tmp)
        
        #df_weight_list.append(df_rank_score_tmp)
        # filter_stk = df_rank_score_tmp.code.iloc[:int(len(df_rank_score_tmp)/3*2)].tolist()
        # stk_filter_list += filter_stk
        # stk_filter_dict[date] = filter_stk
        
#df_weight_del_neg = pd.concat(df_weight_del_neg_list, axis=0) # 剔除负分等权
df_weight_del_neg_rank = pd.concat(df_weight_del_neg_rank_list, axis=0) # 剔除负分加权

def get_weight(df_weight):
    df_weight = df_weight.pivot(index='date', columns='code', values='weight')
    df_weight.fillna(0, inplace=True)
    df_weight.index = pd.to_datetime(df_weight.index, format='%Y%m%d')
    return df_weight

#df_weight_del_neg = get_weight(df_weight_del_neg)
df_weight_del_neg_rank = get_weight(df_weight_del_neg_rank)

# stk_filter_list = set(stk_filter_list)
# df_order_filter = pd.DataFrame(index=df_test.index, columns=stk_filter_list)
# for date in df_test.index:
#     df_order_filter.loc[date, stk_filter_dict[str(date.date()).replace('-','')]]=1
# df_order_filter.fillna(0, inplace=True)
# df_weight_filter = df_order_filter.groupby(axis=0, level=0).apply(lambda x: x / (np.abs(x).sum().sum()))
#df_pnl_del_neg = backtest(df_weight_del_neg, df_next_ret[df_weight_del_neg.columns])

df_pnl_del_neg_rank = backtest(df_weight_del_neg_rank, df_next_ret[df_weight_del_neg_rank.columns])

fig,ax = plt.subplots(2, 1,figsize=(10,8))
df_pnl_equal_weight = pd.read_csv(r'D:\work\GTJA\section\day_equal.csv', names=['date','等权'], index_col=0, parse_dates=True)
df_pnl = pd.concat([df_pnl_equal_weight, df_pnl_del_neg_rank], axis=1)
df_pnl = df_pnl[df_pnl.index <= '20200925']
df_pnl.columns = ['等权买入','按打分加权']
df_pnl.plot();plt.show()

#df_pnl.plot(ax=ax[0])

# hold_num_list = list(map(lambda x:len(x), df_weight_del_neg_list))
# ax2= ax[1]
# ax2.plot(df_pnl.index, hold_num_list)
# ax2.legend(['成交个股数'])
# plt.show()
#
# # 统计打分整体为负时各因子收益率表现
# df_tmp = pd.concat(df_weight_del_neg_rank_list, axis=0)
# neg_date = pd.to_datetime(df_tmp[df_tmp.neg==1].date.unique(), format='%Y%m%d')
# df_test[df_test.index.isin(neg_date)].mean()

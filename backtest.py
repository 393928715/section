import pandas as pd
from zgtools.data_tool import *
import matplotlib.pyplot as plt

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读入每日股票收益
df_stk_mkt = pd.read_csv('data/stk_mkt.csv', parse_dates=['date'])
df_stk_mkt.code = df_stk_mkt.code.apply(lambda x:int(x[:6]))
df_next_ret = df_stk_mkt[['date', 'code', 'next_open2open']].pivot_table(columns='code', index='date', values='next_open2open')

# 读入股票权重矩阵
df_weight = pd.read_csv('expected_return_barra_opt_weight.csv', index_col='date', parse_dates=True)
df_weight.columns = map(lambda x:float(x), df_weight.columns)

# 读入等权买入净值
df_pnl_equal_weight = pd.read_csv(r'D:\work\GTJA\section\day_equal.csv', names=['date','等权'], index_col=0, parse_dates=True)

df_next_ret = df_next_ret[df_next_ret.index.isin(df_weight.index)]
df_pnl_n = backtest(df_weight, df_next_ret[df_weight.columns])
df_pnl_n.name = '按35天动量'

df_pnl = pd.concat([df_pnl_equal_weight, df_pnl_n], axis=1)
df_pnl.dropna(axis=0, inplace=True)
df_pnl.plot();plt.show()
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
import scipy.optimize as sco
from dateutil.relativedelta import relativedelta
from sklearn.covariance import MinCovDet, EmpiricalCovariance
import tushare as ts

ts.set_token('46304a165e1a71a0ff4ffaaa9c3da977498c1d1c918c798322ffe6b1')
pro = ts.pro_api()

df_index_weight = pro.index_weight(index_code='399317.SZ', start_date='20201030', end_date='20201030')


period_adjustment = 1

def objFunc(w, X, f, F):
    C = np.dot(np.dot(X, F), X.T)
    val = 5*np.dot(np.dot(w.T, C), w) - np.dot(w.T,np.dot(X, f))
    return val

def portfolio_expected_return(w, r):
    r_mean = r.mean()  # 各证券的日均收益率
    p_mean = np.sum(r_mean * w) * period_adjustment  # 组合下一期收益率期望
    return -p_mean

def portfolio_covariance(r, method='normal'):
    if method == 'normal':
        r_cov = r.cov() * period_adjustment
    elif method == 'mcd':
        r_cov = MinCovDet(random_state=0).fit(r).covariance_ * period_adjustment
    elif method == 'mest':
        r_cov = EmpiricalCovariance().fit(r).covariance_ * period_adjustment
    return r_cov

def portfolio_risk(r_cov, w):
    p_var = np.dot(w.T, np.dot(r_cov, w))  
    p_std = np.sqrt(p_var)  
    return p_std

# 目标函数 objective function: 风险平价
def risk_parity_obj(w, r, x_t, v_method='normal'):
    r_cov = portfolio_covariance(r, v_method)
    p_std = portfolio_risk(r_cov, w)
    risk_target = np.multiply(p_std, x_t)
    w = np.mat(w)
    mrc = np.dot(r_cov, w.T) / p_std            
    p_rc = np.multiply(mrc, w.T)                
    J = sum(np.square(p_rc - risk_target.T))[0, 0]      
    return J

def get_opt_data(date):
    seri_stk_tmp = df_stk_pool[date].dropna()
    df_barra_tmp = df_barra_zscore[(df_barra_zscore.index.get_level_values(
        'date').isin([date]))
                                   & (df_barra_zscore.index.get_level_values('code').isin(seri_stk_tmp))]
    df_barra_tmp.fillna(0, inplace=True)  # 无值的填充为0，不然后面矩阵运算会报错
    df_next_ret_tmp = df_stk_mkt[(df_stk_mkt.index.get_level_values('date') < date) &
                                 (df_stk_mkt.index.get_level_values('date') >= (
                                             date - relativedelta(days=35))) &
                                 (df_stk_mkt.index.get_level_values('code').isin(seri_stk_tmp))][next_ret_col].unstack()
    return seri_stk_tmp, df_next_ret_tmp, df_barra_tmp

# 目标函数 objective function: 均值方差
def mean_variance_obj(w, r, r_f, v_method='normal'):
    p_mean = portfolio_expected_return(w, r)
    r_cov = portfolio_covariance(r, v_method)
    p_std = portfolio_risk(r_cov, w)
    p_sharpe = (p_mean - r_f) / p_std
    return -p_sharpe

# 读取股票池
df_stk_pool = pd.read_csv('data/stk_pool.csv', header=1)
df_stk_pool.columns = pd.to_datetime(df_stk_pool.columns, format='%Y%m%d')
df_stk_pool = df_stk_pool[df_stk_pool.columns[df_stk_pool.columns >= '2020-01-01']]

# 读取收益率行情数据
df_stk_mkt = pd.read_csv('data/stk_mkt.csv', parse_dates=['date'])
df_stk_mkt.code = df_stk_mkt.code.apply(lambda x:int(x[:6]))
df_stk_mkt.set_index(['date','code'], inplace=True)

# 读取barra因子暴露数据
df_barra_zscore = pd.read_csv('data/barra_zsocre_2019.csv', index_col=['date','code'], parse_dates=['date'])
# df_barra_zscore.columns = ['date','code'] + df_barra_zscore.columns[2:].tolist()
# df_barra_zscore = df_barra_zscore[df_barra_zscore.date >= 20190101]
# df_barra_zscore.to_csv('data/barra_zsocre_2019.csv', index=None)

# 剔除barra里没有的股票
for date in df_stk_pool.columns:
    codes = set(df_stk_pool[date]).intersection(set(df_barra_zscore[df_barra_zscore.index.get_level_values('date').isin([date])].index.get_level_values('code')))
    df_stk_pool[date] = df_stk_pool[df_stk_pool[date].isin(codes)]

date_list = []
pool_list = []
w_list = []

for date in df_stk_pool.columns:
    # 读取2020-03-01
    next_ret_col = 'next_open2open'
    train_interval = 1.5
    
    seri_stk_tmp, df_next_ret_tmp, df_barra_tmp = get_opt_data(date)
    
    if len(df_next_ret_tmp)==0:
        break
    
    num_asset = df_next_ret_tmp.shape[1]
    w0 = 1.0*np.ones(num_asset)/num_asset
    bounds = [(0,0.03) for i in range(num_asset)]
    x_t = [1 / num_asset for _ in range(num_asset)]         # 平均分配风险权重
    # 风格限制矩阵
    barra_cons = 0.01*np.ones(11) 
    #(np.dot(w0,df_barra_tmp))
    cons = ({'type':'eq', 'fun': lambda w: sum(w)-1.0 },)
            #{'type': 'ineq','fun' : lambda w:barra_cons - abs(np.dot(w,df_barra_tmp))},)

    res = minimize(portfolio_expected_return, w0, args=(df_next_ret_tmp),bounds=bounds, constraints=cons)
    
    w_opt = list(res['x'])
    
    date_list.extend([date] * num_asset)
    pool_list.extend(seri_stk_tmp)
    w_list.extend(w_opt)
    print(res['success'], date)

df_w_record = pd.DataFrame({'date': date_list, 'pool': pool_list, 'w': w_list})
df_w = df_w_record.pivot_table(index='date', columns='pool', values='w')  # 生成权重矩阵，用于回测
df_w.to_csv('expected_return_barra_opt_weight.csv')

    # w_opt = list(res['x'])
    # print(abs(np.dot(w_opt,df_barra_tmp)))
    # print((np.dot(w0,df_barra_tmp)))
# result = sco.minimize(objFunc, w0, (X, f, F), method='SLSQP', constraints=cons, bounds=bound,options={'maxiter' : 500})
# w_opt = result.x
# print result.items()[1:5]  #打印求解信息



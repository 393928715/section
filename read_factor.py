import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

def save_obj(obj, name):
    with open('data/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name ):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

root_dir = r'D:\work\GTJA\factors_barra_data'
folder_list = os.listdir(root_dir)

def read_ic():
    folder = folder_list[0]

    df_ic_list = []
    for folder in folder_list:
        fpath = os.path.join(root_dir, folder, 'sectionIndicators', 'factor_indicators_IC.h5')
        if os.path.exists(fpath):
            df_ic_tmp = pd.read_hdf(fpath)['OODay1']
            df_ic_tmp.name = folder # 列名赋值为因子名
            df_ic_list.append(df_ic_tmp)
    df_ic = pd.concat(df_ic_list, axis=1)
    df_ic.to_csv('data/fac_ic.csv')
    return df_ic

def read_pnl():
    df_ic_list = []
    for folder in folder_list:
        fpath = os.path.join(root_dir, folder, 'sectionIndicators', 'factor_indicators_PnL.h5')
        if os.path.exists(fpath):
            df_ic_tmp = pd.read_hdf(fpath)['OODay1']
            df_ic_tmp.name = folder # 列名赋值为因子名
            df_ic_list.append(df_ic_tmp)
    df_ic = pd.concat(df_ic_list, axis=1)
    df_ic.to_csv('data/barra_fac_ret.csv')
    return df_ic

def read_rank_score(type='df'):

    factor_pnl_dict = {}
    df_tmp_list = []
    for folder in folder_list:
        fpath = os.path.join(root_dir, folder, 'scoresMatrix', 'factor_scores_raw.h5')
        if os.path.exists(fpath):
            df_tmp = pd.read_hdf(fpath)#['OODay1']
            #factor_pnl_dict[folder] = df_tmp
            df_tmp = df_tmp.unstack()
            df_tmp = df_tmp.swaplevel()
            df_tmp.name = folder # 列名赋值为因子名
            df_tmp_list.append(df_tmp)
            print(folder)

    df_barra_rank_score = pd.concat(df_tmp_list, axis=1)
    df_barra_rank_score.to_csv('data/barra_rank_score')

    save_obj(factor_pnl_dict,'fac_rank.pkl' )
    df = pd.concat(df_tmp_list, axis=0)
    df_ic.to_csv('data/fac_rank.csv')

    #d = load_obj('fac_rank.pkl')
    return factor_pnl_dict

def get_stk_pool():
    # 读取现有日频因子数据
    root_dir = r'D:\work\GTJA\factor_data\factor_day'
    flist = os.listdir(root_dir)
    df_stk_list = []
    for f in flist:
        fpath = os.path.join(root_dir, f)
        df_factor_socre_tmp = pd.read_hdf(fpath) # 用这个接口有中文路径会报错
        df_stk_pool_tmp = df_factor_socre_tmp.T.groupby(axis=1, level=0).apply(lambda x:x[x.columns[0]].sort_values(ascending=False).iloc[:20].index.to_series().reset_index(drop=True))
        df_stk_list.append(df_stk_pool_tmp)

        print(f)
    df_stk_pool = pd.concat(df_stk_list, axis=0)
    df_stk_pool = df_stk_pool.groupby(axis=1, level=0).apply(lambda x: x.drop_duplicates().reset_index(drop=True))
    df_stk_pool.to_csv('data/stk_pool.csv', index=None)
    return df_stk_pool

def read_factor_day():
    # 读取现有因子日频数据，做打分算股票池
    root_dir = r'D:\work\GTJA\factor_data\factor_day'
    flist = os.listdir(root_dir)
    df_tmp_list = []
    for f in flist:
        fpath = os.path.join(root_dir, f)
        df_factor_socre_tmp = pd.read_hdf(fpath) # 用这个接口有中文路径会报错
        df_factor_socre_tmp = df_factor_socre_tmp.unstack()
        df_factor_socre_tmp = df_factor_socre_tmp.swaplevel()
        df_factor_socre_tmp.name = f  # 列名赋值为因子名
        df_tmp_list.append(df_factor_socre_tmp)
        print(f)

    df_factor_day_rank_score = pd.concat(df_tmp_list, axis=1)
    df_factor_day_rank_score.to_csv('data/factor_day_rank_score.csv')
    # df_stk_pool = df_stk_pool.groupby(axis=1, level=0).apply(lambda x: x.drop_duplicates().reset_index(drop=True))
    # df_stk_pool.to_csv('data/stk_pool.csv', index=None)
    return df_factor_day_rank_score

def read_rank_score():
    df_ic_list = []
    for folder in folder_list:
        fpath = os.path.join(root_dir, folder, 'scoresMatrix', 'factor_scores_rank.h5')
        if os.path.exists(fpath):
            df_ic_tmp = pd.read_hdf(fpath)#['OODay1']
            df_ic_tmp.name = folder # 列名赋值为因子名
            df_ic_list.append(df_ic_tmp)
    df_ic = pd.concat(df_ic_list, axis=1)
    df_ic.to_csv('data/barra_fac_ret.csv')
    return df_ic

def read_barra_zscore():
    df_tmp_list = []
    for folder in folder_list:
        fpath = os.path.join(root_dir, folder, 'scoresMatrix', 'factor_scores_zscore.h5')
        if os.path.exists(fpath):
            df_tmp = pd.read_hdf(fpath)#['OODay1']
            #factor_pnl_dict[folder] = df_tmp
            df_tmp = df_tmp.unstack()
            df_tmp = df_tmp.swaplevel()
            df_tmp.name = folder # 列名赋值为因子名
            df_tmp_list.append(df_tmp)
        print(folder)
    df_barra_zscore = pd.concat(df_tmp_list, axis=1)
    return df_barra_zscore

#df_barra_zscore = read_barra_zscore()
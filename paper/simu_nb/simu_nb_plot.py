
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import scipy as sp
import h5py
from functools import reduce
import matplotlib.lines as mlines

path_base = '/home/jinandmaya/'
sys.path.append(path_base+'methods/')
from metrics import comp_stat, comp_score

path_base = path_base + 'simu_nb/'


def legend_title_left(leg):
    c = leg.get_children()[0]
    title = c.get_children()[0]
    hpack = c.get_children()[1]
    c._children = [hpack]
    hpack._children = [title] + hpack.get_children()


p = 2000
if len(sys.argv)>1:
    ind = str(sys.argv[1])
else:
    ind = ''
r_list = [2,4,6]
method_list = ['wilc', 'DESeq', 'cocoa', 'cinemaot', 'cinemaotw'] \
    + ['ruv_r_{}'.format(r) for r in r_list] \
    + ['ruv3nb_r_{}'.format(r) for r in r_list] \
    + ['causarray_r_{}'.format(r) for r in r_list]
c = 0.1
alpha = 0.1
n_list = [100, 500, 1000, 5000]

df_res = pd.DataFrame()
for n in n_list:
    path_data = path_base+'data/simu_{}{}/'.format(n, ind)
    path_result = path_base+'results/simu_{}{}/'.format(n, ind)

    df = []
    for seed in range(50):
        filename = path_data+'simu_data_{}.h5'.format(seed)
        if not os.path.exists(filename):
            continue
        with h5py.File(filename, 'r') as f:
            theta = np.array(f['theta'])
        print(seed)

        try:
            for method in method_list:
                
                filename = '{}{}_{}.csv'.format(path_result, method, seed)
                if not os.path.exists(filename):
                    print(method)
                    continue
                _df = pd.read_csv(filename, index_col=0)

                if method == 'causarray':
                    rej = (_df['rej']==1).values.astype(int)
                    res = comp_stat(theta, rej, c)
                    res = np.r_[[method.replace('causarray', 'causarray-fdx'), seed], res]
                    df.append(res)
                
                rej = (_df['padj']<alpha).values.astype(int)
                res = comp_stat(theta, rej, c)
                res = np.r_[[method, seed], res]
                df.append(res)
        except:
            continue
        
    df = pd.DataFrame(df, columns = ['method', 'seed', 'typeI_err', 'FDR', 'power', 'FDX', 'num_dis'])
    df[['typeI_err', 'FDR', 'power', 'FDX', 'num_dis']] = df[['typeI_err', 'FDR', 'power', 'FDX', 'num_dis']].astype(float)
    df['n']  = n

    df_res = pd.concat([df_res, df], axis=0)
df_res.reset_index(drop=True, inplace=True)
df_res.to_csv(path_base+'results/result{}_test.csv'.format(ind))

print(df_res.groupby(['n','method'])[['typeI_err', 'FDR', 'power', 'FDX', 'num_dis']].median())








r_list = [2,4,6]
method_list = ['cocoa', 'cinemaot', 'cinemaotw']  \
    + ['ruv_r_{}'.format(r) for r in r_list] \
    + ['ruv3nb_r_{}'.format(r) for r in r_list] \
    + ['causarray_r_{}'.format(r) for r in r_list]
r = 2

df_res = pd.DataFrame()
for n in n_list:
    path_data = path_base+'data/simu_{}{}/'.format(n, ind)
    path_result = path_base+'results/simu_{}{}/'.format(n, ind)

    df = []
    for seed in range(50):
        try:
            filename = path_data+'simu_data_{}.h5'.format(seed)
            if not os.path.exists(filename):
                continue
            with h5py.File(filename, 'r') as f:
                A = np.array(f['A'], dtype='float')
                W = np.array(f['W'], dtype='float')
                Y = np.array(f['Y'], dtype='float')
                Z = W[:,-r:]
                metadata = np.array(f['metadata'], dtype='float')

                celltype = metadata[np.unique(metadata[:,1], return_index=True)[1],-1].astype(int)
            print(seed)
            for method in method_list:
                if method == 'cocoa':
                    W_hat = None
                else:
                    filename = '{}{}_W_{}.csv'.format(path_result, method, seed)
                    if not os.path.exists(filename):
                        print(method, 'W')
                        continue
                    W_hat = pd.read_csv(filename, index_col=0).values
                filename = '{}{}_cf_{}.csv'.format(path_result, method, seed)
                if not os.path.exists(filename):
                    print(method, 'cf')
                    break
                CF = pd.read_csv(filename, index_col=0).values
                res = comp_score(Y, CF, celltype, Z, W_hat)
                res = np.r_[[method, seed], res]

                df.append(res)
        except:
            pass
    df = pd.DataFrame(df, columns = ['method', 'seed', 'ARI', 'ASW'])
    df[['ARI', 'ASW']] = df[['ARI', 'ASW']].astype(float)
    df['n']  = n

    df_res = pd.concat([df_res, df], axis=0)

df_res.reset_index(drop=True, inplace=True)
df_res.to_csv(path_base+'results/result{}_deconfound.csv'.format(ind))
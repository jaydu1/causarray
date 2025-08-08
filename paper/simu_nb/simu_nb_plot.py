
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
method_list = ['wilc', 'DESeq', 'cocoa', 'cinemaot', 'cinemaotw', 'mixscape'] \
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


sns.set(font_scale=1.2)
fig, axes = plt.subplots(1,2, figsize=(10,4), sharex=True, sharey=False)
for j, metric in enumerate(['FDR', 'power']):
    sns.boxplot(data=df_res, x='n', y=metric, hue='method', ax=axes[j])

axes[0].axhline(alpha, color='r', linestyle='--')
lines_labels = [ax.get_legend_handles_labels() for ax in [axes[1]]]
handles, labels = [sum(lol, []) for lol in zip(*lines_labels)]

axes[0].get_legend().remove()
axes[1].get_legend().remove()
legend = fig.legend(handles=handles, labels=labels,
                    loc=9, ncol=5, title=None, frameon=False)
legend_title_left(legend)

fig.tight_layout()
fig.subplots_adjust(top=0.85)

plt.savefig(path_base+'results/simu{}_res.pdf'.format(ind), bbox_inches='tight', pad_inches=0, dpi=300)

print(df_res.groupby(['n','method'])[['typeI_err', 'FDR', 'power', 'FDX', 'num_dis']].median())








r_list = [2,4,6]
method_list = ['cocoa', 'cinemaot', 'cinemaotw', 'mixscape']  \
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





df_test = pd.read_csv(path_base+'results/result{}_test.csv'.format(ind)).rename({'FDR':'FPR', 'power':'TPR'}, axis=1)
df_cf = pd.read_csv(path_base+'results/result{}_deconfound.csv'.format(ind))

r_list = [2,4,6]
method_name = {
    'wilc':'Wilcoxon', 'DESeq':'DESeq2', 'cocoa':'CoCoA', 'cinemaot':'CINEMA-OT', 'cinemaotw':'CINEMA-OT-W', 'mixscape':'Mixscape',
    }
    
method_name = reduce(lambda a, b: dict(a, **b), 
    [{'ruv_r_{}'.format(r):'RUV $r={}$'.format(r) for r in r_list}, 
      {'ruv3nb_r_{}'.format(r):'RUV-III-NB $r={}$'.format(r) for r in r_list}, 
      {'causarray_r_{}'.format(r):'causarray $r={}$'.format(r) for r in r_list}
    ])

df_test = df_test[df_test['method'].isin(method_name.keys())]
df_cf = df_cf[df_cf['method'].isin(method_name.keys())]
df_test['method'] = df_test['method'].map(method_name)
df_cf['method'] = df_cf['method'].map(method_name)

df_test = df_test[df_test['n'].isin(n_list)]
df_cf = df_cf[df_cf['n'].isin(n_list)]

method_list = method_name.values()#df_test['method'].unique()
# palette = sns.color_palette()[:len(method_list)]
palette = reduce(lambda l1, l2: l1+l2, [sns.color_palette(name)[:len(r_list)*2:2] for name in ['Reds', 'Greens', 'Blues']])
hue_order = {i:c for i,c in zip(method_list, palette) }

sns.set(font_scale=1.3)
fig, axes = plt.subplots(1,4, figsize=(16,5), sharex=False, sharey=False)
for j, metric in enumerate(['FPR', 'TPR']):
    sns.boxplot(data=df_test, x='n', y=metric, hue='method', hue_order=hue_order,
        ax=axes[j+2], palette=palette, showfliers=False)

for j, metric in enumerate(['ARI', 'ASW']):
    sns.boxplot(data=df_cf, x='n', y=metric, hue='method', hue_order=hue_order,
        ax=axes[j], palette=palette, showfliers=False)

axes[2].axhline(0.1, color='r', linestyle='--')
lines_labels = [ax.get_legend_handles_labels() for ax in [axes[1]]]
handles, labels = [sum(lol, []) for lol in zip(*lines_labels)]
handles = [mlines.Line2D([], [], linestyle='None')] * 3 + handles[::3] + handles[1::3] + handles[2::3]
labels = ['RUV', 'RUV-III-NB', 'causarray',
            '$r=2$', '$r=2$', '$r=2$',
            '$r=4$', '$r=4$', '$r=4$',
            '$r=6$', '$r=6$', '$r=6$']

for j in range(4):
    axes[j].get_legend().remove()
    axes[j].tick_params(axis='both', which='major', labelsize=10)
    axes[j].set_xlabel('Sample size $n$')
legend = fig.legend(handles=handles, labels=labels,
                    loc=9, ncol=4, title=None, frameon=False)           
legend_title_left(legend)

fig.tight_layout()
fig.subplots_adjust(top=0.78)

plt.savefig(path_base + 'results/simu_nb_r{}.pdf'.format(ind), bbox_inches='tight', pad_inches=0, dpi=300)
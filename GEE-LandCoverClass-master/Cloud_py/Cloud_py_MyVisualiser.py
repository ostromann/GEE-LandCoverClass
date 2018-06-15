import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import Normalize

import numpy as np
import pandas as pd


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))    

def save_accuracy_plot(folder, df):
    #Plotting the the accuracy over number of features
    #Get best mean_train_score per Selection Method and number of features
    df_m = (df.assign(rn=df.sort_values(['mean_test_score'], ascending=False)
                .groupby(['reducer','n_features'])
                .cumcount()+1)
                .query('rn<2')
    )
    reducers = df_m['reducer'].drop_duplicates().tolist()
    COLORS = 'bgrcmyk'

    plt.figure('accuracy', figsize =(7,5))
    for color,reducer_label in zip(COLORS,reducers):
        if reducer_label != '': #Labels to exclude!
            reducer = df_m[df_m['reducer'] == reducer_label][['n_features','mean_test_score','std_test_score']]
            reducer = reducer.sort_values('n_features')

            #----full line for mean
            plt.plot(reducer['n_features'],reducer['mean_test_score'], '-', label=reducer_label, c= color)  
            #----filled area for std
            plt.fill_between(x=reducer['n_features'], 
                             y1=reducer['mean_test_score']-reducer['std_test_score'], 
                             y2=reducer['mean_test_score']+reducer['std_test_score'],
                            color=color, alpha =0.2, interpolate = True)


    plt.legend(loc='best')
    plt.title('Performance of Different Feature Extraction Methods')
    plt.xlabel('Number of Features')
    plt.ylabel('Mean Accuracy')
    #plt.xscale('log', basex = 2)
    plt.grid(b=True, which='major', linestyle='-', alpha=0.6)
    plt.grid(b=True, which='minor', linestyle='--', alpha=0.3)
    plt.minorticks_on()
    
    plt.savefig(folder + 'accuracy.png')
    
def save_linear_parameter_search(folder, df):
    df = df[df['param_classify__kernel'] == 'linear']
    #Get best number of features per selection method
    best_opt = (df.assign(rn=df.sort_values(['mean_train_score'], ascending=False)
                .groupby(['reducer'])
                .cumcount()+1)
                .query('rn<2')
    )[['reducer','n_features']]

    zipper = zip(best_opt['reducer'].tolist(), best_opt['n_features'])
    n_cols = 3
    n_rows = len(best_opt['reducer'].tolist())%n_cols+1

    plt.figure('linear',figsize=(16,5))
    # plot parameter searches per best number of features per selection method
    for k,(reducer,n_features) in enumerate(zipper):
        best_linear = df[(df['reducer'] == reducer) & (df['n_features'] == n_features) & (df['param_classify__kernel'] == 'linear')]

        plt.figure('linear')
        plt.subplot(n_rows,n_cols,k+1)
        plt.title(reducer)
        plt.xlabel('C')
        plt.xscale('log')
        plt.ylabel('Mean Accuracy')
        plt.grid(b=True, which='major', linestyle='-', alpha=0.6)
        plt.grid(b=True, which='minor', linestyle='--', alpha=0.3)
        plt.minorticks_on()
        plt.plot(best_linear['param_classify__C'],best_linear['mean_test_score'], '-',c= 'b')
        plt.fill_between(x=best_linear['param_classify__C'],
                         y1=best_linear['mean_test_score']-best_linear['std_test_score'],
                         y2=best_linear['mean_test_score']+best_linear['std_test_score'],
                         alpha =0.2, interpolate = True)

    plt.savefig(folder + 'linear_search.png')
    
def save_rbf_parameter_search(folder, df):
    #Get best number of features per selection method
    df = df[df['param_classify__kernel'] == 'rbf']
    best_opt = (df.assign(rn=df.sort_values(['mean_train_score'], ascending=False)
                .groupby(['reducer'])
                .cumcount()+1)
                .query('rn<2')
    )[['reducer','n_features']]

    zipper = zip(best_opt['reducer'].tolist(), best_opt['n_features'])
    n_cols = 3
    n_rows = len(best_opt['reducer'].tolist())%n_cols+1

    C_OPTIONS = df['param_classify__C'].drop_duplicates().tolist()
    GAMMA_OPTIONS = df['param_classify__gamma'].drop_duplicates().dropna(axis=0).tolist()

    plt.figure('rbf',figsize=(16,5))
    # plot parameter searches per best number of features per selection method
    for k,(reducer,n_features) in enumerate(zipper):
        best_rbf = df[(df['reducer'] == reducer) & (df['n_features'] == n_features) & (df['param_classify__kernel'] == 'rbf')]

        scores = best_rbf['mean_test_score'].values.reshape(len(C_OPTIONS),len(GAMMA_OPTIONS))

        plt.figure('rbf')
        plt.subplot(n_rows,n_cols,k+1)
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot, norm=MidpointNormalize(midpoint=df['mean_test_score'].max()*0.95))
        plt.title(reducer)
        plt.xlabel('gamma')
        plt.ylabel('C')
        plt.colorbar()
        plt.xticks(np.arange(len(GAMMA_OPTIONS)), GAMMA_OPTIONS, rotation=45)
        plt.yticks(np.arange(len(C_OPTIONS)), C_OPTIONS)

    plt.savefig(folder + 'RBF.png')



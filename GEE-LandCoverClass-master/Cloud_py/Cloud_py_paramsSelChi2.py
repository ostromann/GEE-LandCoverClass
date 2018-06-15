import numpy as np
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.externals import joblib
from sklearn.externals.joblib import Memory

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, NMF, FastICA, IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, RFE, VarianceThreshold
from sklearn.svm import SVC
from matplotlib.colors import Normalize
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import datetime

def initialise_pipe_param_grid(n_features):
    now = datetime.datetime.now()
    print('started at:',now.strftime("%Y-%m-%d_%H-%M"))
    cachedir = mkdtemp(prefix='py_sklearn_tmp')
    memory = Memory(cachedir=cachedir, verbose=0)
    print('cache directory created at:', cachedir)

    #Create a scikit-learn pipeline
    pipe = Pipeline([('variance_thresh', VarianceThreshold()), #remove constant features
                     ('scale',MinMaxScaler()),#one standardisation step
                     ('reduce_dim', SelectKBest(mutual_info_classif)), #one transformation step
                     ('classify', SVC())], #one classification step
                     memory=memory)
                     
    N_FEATURES = [1, 2, 3, 4, 5, 6, 7, 9, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 125, 160, 200, 250, 325,400, 475, 550,650,700]

    param_grid = [
#        {
#            'reduce_dim': [SelectKBest(f_classif)],
#            'reduce_dim__k': N_FEATURES,
#            'classify__C': np.logspace(-3,1,5),
#            'classify__kernel': ['linear'],
#        }, 
        {
            'reduce_dim': [SelectKBest(chi2)],
            'reduce_dim__k': N_FEATURES,
            'classify__C': np.logspace(-1,6,8),
            'classify__kernel': ['rbf'],
            'classify__gamma': np.logspace(-6,0,7)
        }, 
    ]
    
    print('Parameter Grid:\n', param_grid)
    print('Pipeline:\n', pipe)
    return (pipe, param_grid, now, cachedir)

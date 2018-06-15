import numpy as np
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.externals.joblib import Memory
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, NMF, FastICA, IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from matplotlib.colors import Normalize
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import datetime

def initialise_pipe_param_grid():
    now = datetime.datetime.now()
    print('started at:',now.strftime("%Y-%m-%d_%H-%M"))
    cachedir = mkdtemp(prefix='py_sklearn_tmp')
    memory = Memory(cachedir=cachedir, verbose=0)
    print('cache directory created at:', cachedir)

    #Create a scikit-learn pipeline
    pipe = Pipeline([('scale',RobustScaler()),#one standardisation step
                     ('reduce_dim', LDA()), #one transformation step
                     ('classify', SVC())], #one classification step
                     memory=memory)
    N_FEATURES = list(sorted(set(list(map(lambda x: int(round(x)), np.logspace(-3,-1,15)*700)))))

    param_grid = [
        {
            'reduce_dim': [PCA()],
            'reduce_dim__n_components': N_FEATURES,
            'classify__C': np.logspace(-4,2,7),
            'classify__kernel': ['linear']
        },
        {
            'reduce_dim': [PCA()],
            'reduce_dim__n_components': N_FEATURES,
            'classify__C': np.logspace(-2,6,9),
            'classify__kernel': ['rbf'],
            'classify__gamma': np.logspace(-8,0,9)
        }, 
    ]
    
    print('Parameter Grid:\n', param_grid)
    print('Pipeline:\n', pipe)
    return (pipe, param_grid, now, cachedir)


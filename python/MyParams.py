"""Module for creating a Pipeline and a parameter grid for grid search.


    author: Oliver Stromann
    created: 2018-11-19
    last updated: 2018-11-19
"""
import numpy as np
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.externals import joblib
from sklearn.externals.joblib import Memory

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier


class Params:
    '''Class for constructing the Parameter Grid for the grid search.

    '''
      
    def __init__(self, option,*, n_features = 700, n_classes = 10, verbose = 0):
        """Instantiate a Pipeline and Parameter Grid.

        Args:
            option (str): 'None', 'LDA', 'MI', 'F_Score'. Select different Dimensionality reduction method.
            n_features(int): Maximal number of features.
            n_classes (int): Number of unique class labels.
            verbose (int): Level of verbosity for cachedir creation.

        """
        self.options = {'None': default_svm,
                        'LDA': LDA,
                        'MI': MI,
                        'F_Score': F_Score,
        }
        self.n_features = n_features
        self.n_classes = n_classes
        self.verbose = verbose

        #create cache directory for storing pipelines 
        self.cachedir = mkdtemp(prefix='py_sklearn_tmp')
        self.memory = memory = Memory(cachedir=self.cachedir, verbose=self.verbose)
        
        self.pipeline, self.param_grid = self.options[option]()      
        
    def default_svm(self):
        pipeline = Pipeline([('variance_thresh', VarianceThreshold()),
                             ('scale',RobustScaler()),
                             #('reduce_dim', FastICA()),
                             ('classify', OneVsRestClassifier(SVC()))
                            ],
                            memory=self.memory)
        param_grid = [
            {
                #'reduce_dim': [FastICA()],
                #'reduce_dim__n_components': self.n_features,
                'classify__estimator__C': np.logspace(3,11,9),
                'classify__estimator__kernel': ['rbf'],
                'classify__estimator__gamma': np.logspace(-12,-4,9)
            }, 
        ]
        return (pipeline, param_grid)
    
    def MI(self):
        pipeline = Pipeline([('variance_thresh', VarianceThreshold()),
                             ('scale',RobustScaler()),
                             ('reduce_dim', SelectKBest(mutual_info_classif)),
                             ('classify', OneVsRestClassifier(SVC()))
                            ],
                            memory=self.memory)
        
        n_features_list = list(sorted(set(list(map(lambda x: int(round(x)), np.logspace(-3,-0.5,25)*self.n_features))))) 
        + list(sorted(set(list(map(lambda x: int(round(x)),np.logspace(-0.4,0,7)*n_features)))))
        
        param_grid = [
            {
                'reduce_dim': [SelectKBest(mutual_info_classif)],
                'reduce_dim__k': n_features_list,
                'classify__estimator__C': np.logspace(3,8,6),
                'classify__estimator__kernel': ['rbf'],
                'classify__estimator__gamma': np.logspace(-9,-4,6)
            }, 
        ]        
        return (pipeline, param_grid)
    
    def F_Score(self):
        pipeline = Pipeline([('variance_thresh', VarianceThreshold()),
                                   ('scale',RobustScaler()),
                                   ('reduce_dim', SelectKBest(f_classif)), 
                                   ('classify', OneVsRestClassifier(SVC()))
                                  ],
                                  memory=self.memory)
        n_features_list = list(sorted(set(list(map(lambda x: int(round(x)), np.logspace(-3,-0.5,25)*self.n_features))))) 
        + list(sorted(set(list(map(lambda x: int(round(x)),np.logspace(-0.4,0,7)*n_features)))))
        
        param_grid = [
            {
                'reduce_dim': [SelectKBest(f_classif)],
                'reduce_dim__k': n_features_list,
                'classify__estimator__C': np.logspace(2,8,7),
                'classify__estimator__kernel': ['rbf'],
                'classify__estimator__gamma': np.logspace(-10,-3,8)
            }, 
        ]
        return (pipeline, param_grid)
    
    def LDA(self):
        pipeline = Pipeline([('variance_thresh', VarianceThreshold()),
                                   ('scale',RobustScaler()),
                                   ('reduce_dim', LDA()), 
                                   ('classify', OneVsRestClassifier(SVC()))
                                  ],
                                  memory=self.memory)
        n_features_list = list(map(lambda x: int(round(x)),np.linspace(1,self.n_classes-1,self.n_classes-1)))
        
        param_grid = [
            {
                'reduce_dim': [LDA],
                'reduce_dim__n_components': n_features_list,
                'classify__estimator__C': np.logspace(2,8,7),
                'classify__estimator__kernel': ['rbf'],
                'classify__estimator__gamma': np.logspace(-10,-3,8)
            }, 
        ]
        return (pipeline, param_grid)

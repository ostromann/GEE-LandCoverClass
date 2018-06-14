#---IMPORT STATEMENTS---
import MySegments
import MyVisualiser
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit as SSS
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA, NMF, FastICA, IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import VarianceThreshold

from tempfile import mkdtemp
from shutil import rmtree
from sklearn.externals.joblib import Memory
from sklearn.externals import joblib
import datetime

#RBF Parameters heatmap
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

from os import mkdir
from os import listdir
from os.path import isfile, join

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


selection_methods = [chi2, mutual_info_classif, f_classif]
labels =  ['chi2', 'mutual information', 'F-scores']
scalers = [MinMaxScaler(), RobustScaler(), RobustScaler()]

for selection_method, label,scaler in zip(selection_methods,labels,scalers):
    print(selection_method, label, scaler)
    plt.

    mypathes = ['Shapes','S1','S2','S1 VV GLCM', 'S1 VH GLCM', 'S2 GLCM']
    for mypath in mypathes:
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        print('mypath', mypath)
        print(onlyfiles)

        #Get labelled data
        sc = MySegments.SegmentCollection(folder = mypath, segments_path=onlyfiles, labels_path = 'labels.csv', classes_path='class_names.csv')
        idx,X,y = sc.get_labelled()

        X = VarianceThreshold().fit_transform(X,y)
        feature_names = sc.feature_names

        #split train and test data, keep class prior
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify = y)

        # Create a temporary folder to store the transformers of the pipeline
        now = datetime.datetime.now()
        print('started at:',now.strftime("%Y-%m-%d_%H-%M"))

       

        # Create and fit selector
        selector = SelectKBest(selection_method, k='all')
        selector.fit(scaler.fit_transform(X_train), y_train)

        mask = selector.get_support() #list of booleans
        features = [] # The list of your K best features
        scores = []

        for bool, feature, score in zip(mask, feature_names, selector.scores_):
                features.append(feature)
                scores.append(score)

        df = pd.DataFrame()
        df['feature'] = features
        df['score'] = scores
        df = df.sort_values(by='score', ascending=False)

        plt.plot(np.linspace(1, df['feature'].count(),df['feature'].count()),df['score'].tolist(), '-', label=mypath)

    plt.legend(loc='best')
    plt.title(label + ' - Normalized Feature Selection Scores')
    plt.xlabel('Feature')
    plt.ylabel('Relevance')
    plt.xscale('log', basex = 2)
    plt.grid(b=True, which='major', linestyle='-', alpha=0.6)
    plt.grid(b=True, which='minor', linestyle='--', alpha=0.3)
    plt.minorticks_on()
    plt.savefig('fs_cross/'label + '.png')
    plt.clf()
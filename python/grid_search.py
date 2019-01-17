#!/usr/bin/env python
"""Perform classifier optimisation with exhaustive grid search.

Perform the classifier optimisation with exhaustive grid search
and cross-validation. Save the best classifier as persistent model,
save the whole cross-validation result as CSV file.

Example:
        $ python grid_search.py

Todo:
    * Use RandomizedSearch instead
    * Use Commandline arguments for Constants
    * Catch Errors
"""
import MySegments
import MyParams
import MyVisualiser

import numpy as np
import pandas as pd
import datetime

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.externals.joblib import Memory
from sklearn.externals import joblib

from os import mkdir, listdir
from os.path import isfile, join
from tempfile import mkdtemp
from shutil import rmtree

#GLOBALS
MYPATH = '../Beijing/aggregate'
LABELS_PATH = '../Beijing/labels_FID.csv'
CLASSES_PATH = '../class_names.csv'
DR_OPTION = 'Default_SVM' #'LDA' #'MI' #'F_Score' #Dimensionality reduction option
TEST_SPLIT_RANDOM_STATE = 5
TEST_SIZE = 0.3
RSKF_N_SPLITS = 3
RSKF_N_REPEATS = 3
RSKF_RANDOM_STATE= 1

def save_file(pandas_obj, save_folder, filename, *, description ='file', index=False):
    """Saves the pandas object as csv file.

    Args:
         pandas_obj (:obj:): Pandas DataFrame object.
         save_folder (str): Path to folder.
         filename (str): Filename.
         description (str, optional): Description to print on screen.
         index (boolean, optional): Store index or not.

    Returns:
        Nothing
    """
    pandas_obj.to_csv(save_folder + '/' + filename, index=index) 
    print('\n', description, ' saved at:', filename)
    

def main():
    now = datetime.datetime.now()
    print('started grid search with ', DR_OPTION,' at:',now.strftime("%Y-%m-%d_%H-%M"))
    
    #get list of files in MYPATH
    files = [f for f in listdir(MYPATH) if isfile(join(MYPATH, f))]

    #read in CSV files as SegmentCollection
    segment_collection = MySegments.SegmentCollection(folder = MYPATH, segments_path=files, 
                                                      labels_path=LABELS_PATH, classes_path=CLASSES_PATH)
    idx,X,y = segment_collectionc.get_labelled()

    #TODO: check if this is still necessary when it is already in the pipeline
    X = VarianceThreshold().fit_transform(X)

    #split train and test data, maintaining class distribution
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=TEST_SPLIT_RANDOM_STATE, stratify=y)

    print('\nTraining set dimensions: ', np.shape(X_train))

    #configure cross-validation setting
    rskf = RepeatedStratifiedKFold(n_splits=RSKF_N_SPLITS, n_repeats=RSKF_N_REPEATS, random_state = RSKF_RANDOM_STATE)

    #get pipeline and parameter grid
    params = MyParams.Params(option='None', n_features=np.shape(X)[1], n_classes=10)

    #set the scoring metrics
    scoring = {'Accuracy': make_scorer(accuracy_score), 'Kappa': make_scorer(cohen_kappa_score)}
    #choose refit metric for Gridsearch
    refit = 'Accuracy'

    #define grid search with cross validation over given parameters
    grid = GridSearchCV(params.pipeline, cv=rskf, n_jobs=-1, param_grid=params.param_grid, scoring=scoring, 
                        refit=refit, return_train_score = True, verbose=1)

    #perform grid search
    grid.fit(X_train, y_train)

    #save the best estimator
    save_folder = '../results_' + now.strftime("%Y-%m-%d_%H-%M") + '/'
    mkdir(save_folder)

    #store gridSearch parameters for repeatability
    print(grid, file=open(save_folder + 'grid.txt','a'))

    #store best estimator
    best_filename = save_folder + 'best_estimator_refit_' + refit +'.pkl'
    joblib.dump(grid.best_estimator_, best_filename)
    rmtree(params.cachedir) #delete the temporary chache for the transformers
    print('\nbest estimator model saved at:', best_filename)

    #store the cross validation results in a CSV file 
    df = pd.DataFrame(data=grid.cv_results_)
    save_file(df, 'raw results', save_folder + 'raw_results.csv')

    #construct and store readable cross-validation results for the different methods
    if option in set(['None', 'Default_SVM']):
        df['reducer'] = 'NaN'
        df['n_features'] = 'NaN'
        
        df['param_classify__estimator__C']=pd.to_numeric(df['param_classify__estimator__C']) 
        df['param_classify__estimator__gamma']=pd.to_numeric(df['param_classify__estimator__gamma'])
        
        save_file(df, 'cross validation results', save_folder + 'cv_results.csv')
        
    elif option in set(['LDA']):
        df['reducer'] = df.apply(lambda row: str(row['param_reduce_dim']).split('(')[0],axis=1)
        df['n_features']=pd.to_numeric(df['param_reduce_dim__n_components'])
        
        df['param_classify__estimator__C']=pd.to_numeric(df['param_classify__estimator__C']) 
        df['param_classify__estimator__gamma']=pd.to_numeric(df['param_classify__estimator__gamma'])

        save_file(df, 'cross validation results', save_folder + 'cv_results.csv')
        
    elif option in set(['MI', 'F_Score']):
        df['reducer'] = df.apply(lambda row: str(row['param_reduce_dim']).split('(')[0],axis=1) 
        df['reducer'] = df.apply(lambda row: ''.join([row['reducer'],' ',str(row['param_reduce_dim']).split('<function ')[1].split(' at')[0]]),axis=1)
        df['n_features']=pd.to_numeric(df['param_reduce_dim__k'])
        
        df['param_classify__estimator__C']=pd.to_numeric(df['param_classify__estimator__C']) 
        df['param_classify__estimator__gamma']=pd.to_numeric(df['param_classify__estimator__gamma'])

        save_file(df, 'cross validation results', save_folder + 'cv_results.csv')        
        
    else:
        print('invalid option no results stored')
    


    #MyVisualiser.save_accuracy_plot(save_folder, df)
    #MyVisualiser.save_linear_parameter_search(save_folder, df)
    #MyVisualiser.save_rbf_parameter_search(save_folder, df)
    
    
if __name__ == '__main__':
    main()
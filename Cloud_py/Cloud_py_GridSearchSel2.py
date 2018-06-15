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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA, NMF, FastICA, IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import RFE

from tempfile import mkdtemp
from shutil import rmtree
from sklearn.externals.joblib import Memory
from sklearn.externals import joblib
import datetime

#RBF Parameters heatmap
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

from os import mkdir
from os import listdir
from os.path import isfile, join
mypath = 'all'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
#Get labelled data
sc = MySegments.SegmentCollection(folder = mypath, segments_path=onlyfiles, labels_path = 'labels.csv', classes_path='class_names.csv')
idx,X,y = sc.get_labelled()

from sklearn.feature_selection import VarianceThreshold

X = VarianceThreshold().fit_transform(X)


#split train and test data, maintaining class distributions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify = y)

rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state =1)

#from the train data take only 50% -> 40% of total
#X_train, X_dump, y_train, y_dump2 = train_test_split(X_train, y_train, test_size=0.5, random_state=1, stratify = y_train)

import paramsSelAnova2 as MyParams

pipe, param_grid, now, cachedir = MyParams.initialise_pipe_param_grid(np.shape(X)[1])

#perform grid search with cross validation over given parameters
grid = GridSearchCV(pipe, cv=rskf, n_jobs=-1, param_grid=param_grid, return_train_score = True, verbose=1)
grid.fit(X_train, y_train)

#save the best estimator
savefolder = now.strftime("%Y-%m-%d_%H-%M") + '/'
mkdir(savefolder)

#Store gridSearch parameters for repeatability
print(grid, file=open(savefolder + 'grid.txt','a'))

#Store best estimator
best_filename = savefolder + 'best_estimator.pkl'
joblib.dump(grid.best_estimator_, best_filename)
print('best estimating model saved at:', best_filename)

#store the cross validation results in a CSV file 
results_filename = savefolder + 'results.csv'
df = pd.DataFrame(data=grid.cv_results_)

#Construct extra text field for reducer name 
df['reducer'] = df.apply(lambda row: str(row['param_reduce_dim']).split('(')[0],axis=1) 
df['reducer'] = df.apply(lambda row: ''.join([row['reducer'],' ',str(row['param_reduce_dim']).split('<function ')[1].split(' at')[0]]) if row['reducer'] == 'SelectKBest' else row['reducer'],axis=1)
#df['reducer'] = 'NaN'
#df['n_features'] = 'NaN'

#Construct extra field number of features (neccessary, since different reducers have different parameter names) 
df['n_features']=pd.to_numeric(df['param_reduce_dim__k'])#.fillna(df['param_reduce_dim__k']) 
df['param_classify__C']=pd.to_numeric(df['param_classify__C']) 
df['param_classify__gamma']=pd.to_numeric(df['param_classify__gamma'])
#df['param_classify__gamma'] = 'NaN'

#save file 
df.to_csv(results_filename, sep='\t') 
print('cross validation results saved as:', results_filename)

MyVisualiser.save_accuracy_plot(savefolder, df)
MyVisualiser.save_linear_parameter_search(savefolder, df)
MyVisualiser.save_rbf_parameter_search(savefolder, df)

# Delete the temporary cache before exiting
rmtree(cachedir)

import MySegments
import MyVisualiser
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.externals import joblib

import os
from os import path
from os import mkdir
from os import listdir
from os.path import isfile, join


import itertools
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from matplotlib.ticker import ScalarFormatter

def plot_learning_curve(model, train_sizes, train_scores, valid_scores, ax):
  #  plt.figure('train_size')
    ax.plot(train_sizes,np.mean(valid_scores, axis = 1), '-', label =model)

def plot_learning_curve_reduced(model, train_sizes, train_scores, valid_scores, ax):
  #  plt.figure('train_size')
    ax.plot(train_sizes,np.mean(valid_scores, axis = 1), '--', c='black', alpha=0.7, label =model)
    
import datetime
now = datetime.datetime.now()


models = ['None','LDA', 'ICA', 'Chi2', 'mut_inf', 'ANOVA']
model_labels = ['default SVM', 'LDA', 'ICA','Chi Square', 'Mutual Information', 'F-Score']
clfs = []

#load best estimators and cv results
for model in models:    
    for file in os.listdir('./TOST_Results/'+ model):
        if file.endswith('.pkl'):
            #print(os.path.join('./Results/' + model, file))
            clfs.append(joblib.load(os.path.join('./TOST_Results/' + model, file)))
            print(joblib.load(os.path.join('./TOST_Results/' + model, file)))

#Load Data
mypath = 'all'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
#Get labelled data
sc = MySegments.SegmentCollection(folder = mypath, segments_path=onlyfiles, labels_path = 'labels.csv', classes_path='class_names.csv')
idx,X,y = sc.get_labelled()

#split train and test data, maintaining class distributions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify = y)

print('----Learning Curves----')
plt.clf()
fig, ax = plt.subplots(nrows=1,ncols=1, num='learning_curves', figsize=(5,4))
ax.grid(b=True, which='major', linestyle='-', alpha=0.6)
ax.grid(b=True, which='minor', linestyle='--', alpha=0.3)
ax.minorticks_on()
ax.set_ylim(ymin=0.17, ymax=1.03)    

for i,(model,model_label, clf) in enumerate(zip(models, model_labels, clfs)):
    
    # compute learnung curve
    train_sizes, train_scores, valid_scores = learning_curve( 
        clf, X_train, y_train, train_sizes=np.logspace(-1.5, 0,8), cv=7, n_jobs=-1)
    
     # plot learning curves
    if i == 0:
        plot_learning_curve_reduced(model_label, train_sizes/np.shape(X_train)[0], train_scores, valid_scores, ax)
    else:
        plot_learning_curve(model_label, train_sizes/np.shape(X_train)[0], train_scores, valid_scores, ax)
        
    
    ax.legend(loc='lower right')    
    ax.set_xscale('log')
    ax.set_xticks([0.02,0.05,0.1,0.2,0.5,1.0])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    
    ax.set_title('Learning curves')
    ax.set_xlabel('Train-test-ratio')
    ax.set_ylabel('Accuracy')  
    
    fig.tight_layout()
plt.savefig('learning_curves_' + now.strftime("%Y-%m-%d_%H-%M") + '.png')
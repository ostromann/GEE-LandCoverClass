#!/usr/bin/env python
"""Compare different persistent learning MODELS (*.pkl).
Save plotted learning curves and normalised confusion matrices as png.
Save prediction of all segments as csv ([segment_ID, predicted_label]).

Example:
        $ python3 pkl_compare.py

Returns:
        Nothing
"""
import MySegments as MySegments
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.externals import joblib

import datetime
import os
from os import path
from os import mkdir
from os import listdir
from os.path import isfile, join

import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.ticker import ScalarFormatter

#GLOBALS
MODELS = ['F_Score']#,'LDA', 'MI']
MODEL_LABELS = ['F-Score']#,'LDA', 'MI']
FILE_PATH = '../Beijing/0_margin_results/' #folder to subfolders of MODELS containing *.pkl files
SEGMENTS_PATH = '../Beijing/aggregate'
LABELS_PATH = '../Beijing/labels_FID.csv'
CLASSES_PATH = '../class_names.csv'

def plot_learning_curve(model, train_sizes, train_scores, valid_scores, ax):
    """
    This function prints and plots the learning curves.
    """
    ax.grid(b=True, which='major', linestyle='-', alpha=0.6)
    ax.grid(b=True, which='minor', linestyle='--', alpha=0.3)
    ax.minorticks_on()
    ax.set_ylim(ymin=0.10, ymax=1.03)    
    
    ax.plot(train_sizes,np.mean(train_scores, axis = 1), '-', label ='training scores')
    ax.fill_between(x=train_sizes,
                         y1=np.mean(train_scores, axis = 1)-np.std(train_scores, axis = 1)*3,
                         y2=np.mean(train_scores, axis = 1)+np.std(train_scores, axis = 1)*3, 
                         alpha =0.2, interpolate = True)
    
    ax.plot(train_sizes,np.mean(valid_scores, axis = 1), '-', label ='testing scores')
    ax.fill_between(x=train_sizes,
                         y1=np.mean(valid_scores, axis = 1)-np.std(valid_scores, axis = 1)*3,
                         y2=np.mean(valid_scores, axis = 1)+np.std(valid_scores, axis = 1)*3,
                         alpha =0.2, interpolate = True)
    
    ax.legend(loc='lower right')    
    ax.set_xscale('log')
    ax.set_xticks([0.02,0.05,0.1,0.2,0.5,1.0])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    
    ax.set_title('Learning curve - ' + model)
    ax.set_xlabel('Train-test-ratio')
    ax.set_ylabel('Overall Accuracy')  

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.imshow(cm*100 if normalize else cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.0f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]*100, fmt) if normalize else format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#start
def main(unseen_performance = False, learning_curves = False, confusion_matrix = False, prediction = False, save = False):
    now = datetime.datetime.now()

    clfs = []
    cv_results = []

    #load best estimators and cv results
    for model in MODELS:    
        for file in os.listdir(FILE_PATH + model):
            if file.endswith('.pkl'):
                #print(os.path.join('./Results/' + model, file))
                clfs.append(joblib.load(os.path.join(FILE_PATH + model, file)))
                #print(joblib.load(os.path.join('./Results_CV9/' + model, file)))
            if file.endswith('.csv'):
                cv_results.append(pd.read_csv(os.path.join(FILE_PATH + model, file), sep='\t'))#, header=1))

    only_files = [f for f in listdir(SEGMENTS_PATH) if isfile(join(SEGMENTS_PATH, f))]

    #Get labelled data
    sc = MySegments.SegmentCollection(folder = SEGMENTS_PATH, segments_path=only_files, labels_path = LABELS_PATH, classes_path=CLASSES_PATH)
    idx,X,y = sc.get_labelled()

    #split train and test data, maintaining class distributions
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5, stratify = y)

    print('\nTraining set size: ', np.shape(X_train), ' Training labels: ', np.shape(y_train))
    print('Test set size: ', np.shape(X_test), ' Test labels: ', np.shape(y_test))

    print('----GridSearch Cross-Validation Score----')
    for model, clf, cv_result in zip(MODELS, clfs, cv_results): 
        print(model)#, clf)
        print('\tAccuracy:\t', cv_result.sort_values(['rank_test_Accuracy'], ascending=True)['mean_test_Accuracy'].head(1).values[0])
        print('\tKappa:\t\t', cv_result.sort_values(['rank_test_Kappa'], ascending=True)['mean_test_Kappa'].head(1).values[0])

    if unseen_perf:
        print('\n\n----Performance on Unseen Data----')
        for i,(model, clf, cv_result) in enumerate(zip(MODELS, clfs, cv_results)):
            #clf.set_params(classify__estimator__verbose = True)
            clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            print(model)
            print('\tAccuracy:\t', accuracy_score(y_pred, y_test))
            print('\tKappa:\t\t', cohen_kappa_score(y_pred, y_test))  

    if learning_curves:
        print('----Learning Curves----')
        for i,(model, clf, cv_result) in enumerate(zip(MODELS, clfs, cv_results)):
            plt.clf()
            fig, ax = plt.subplots(nrows=1,ncols=1, num='learning_curves', figsize=(5,4))
           # compute learnung curve
            train_sizes, train_scores, valid_scores = learning_curve(clf, X_train, y_train, train_sizes=[0.02, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5, 0.8], cv=6, n_jobs=-1)

            #store the learning curve results in csv
            lc = pd.DataFrame()
            lc['train_sizes'] = train_sizes
            lc['mean_train_scores'] = np.mean(train_scores, axis=1)
            lc['mean_valid_scores'] = np.mean(valid_scores,axis=1)
            lc['std_train_scores'] = np.std(train_scores, axis=1)
            lc['std_valid_scores'] = np.std(valid_scores,axis=1)
            lc.to_csv(model + '_learning_curve.csv')     

            # plot learning curves
            plot_learning_curve(model, train_sizes/np.shape(X_train)[0], train_scores, valid_scores, ax)
            fig.tight_layout()
            if save:
                plt.savefig(model + '_learning_curves_' + now.strftime("%Y-%m-%d_%H-%M") + '.png')
            else:
                plt.show()

    if confusion_matrix:
        print('----Confusion Matrices----')
        for i,(model, label, clf, cv_result) in enumerate(zip(MODELS, MODEL_LABELS, clfs, cv_results)):
            plt.clf()
            plt.figure('confusion_matrices', figsize=(5,4))
            # Compute confusion matrix
            cnf_matrix = confusion_matrix(y_test,clf.fit(X_train,y_train).predict(X_test) )
            np.set_printoptions(precision=2)

            # Plot normalized confusion matrix
            plot_confusion_matrix(cnf_matrix, classes=['HDB', 'LDB','R', 'UGS', 'GC', 'AG', 'F', 'W', 'BR', 'WL'], normalize=True,
                                  title='Normalized confusion matrix - ' + label)
            plt.tight_layout()
            if save:
                plt.savefig(model + '_confusion_matrices_' + now.strftime("%Y-%m-%d_%H-%M") + '.png')
            else:
                plt.show()

    if prediction:
        print('----full prediction----')
        idx_train, X_train,y_train = sc.get_labelled()
        idx_test, X_test, y_test = sc.get_unlabelled()
        idx_all, X_all, y_all = sc.get_all()
        print('train data: ', np.shape(X_train))
        print('all data: ', np.shape(X_all))

        import datetime
        now = datetime.datetime.now()

        for i,(model, clf, cv_result) in enumerate(zip(MODELS, clfs, cv_results)):
            outframe = pd.DataFrame()
            outframe['FID'] = idx_all
            outframe['predicted_label'] = clf.fit(X_train,y_train).predict(X_all)


            filename='prediction_1_' + model + '_' + now.strftime("%Y-%m-%d_%H-%M") + '.csv' 
            outframe.to_csv(filename,index=False) 
            print('prediction saved to: ', filename)
              
    
if __name__ == '__main__':
    main(unseen_performance = True, learning_curves = True, confusion_matrix = True prediction = True, save = True):

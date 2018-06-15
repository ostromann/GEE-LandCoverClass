#Module for segments
#classes for Segments and SegmentCollections
#author: Oliver Stromann
#created: 2018-02-13
#last updated: 2018-02-212

#change log
# 2018-02-13: created classes Segment and SegmentCollection
# 2018-02-14: added get functions, added import from CSV, 
#             added splitting of SegmentCollection for training and validation set
# 2018-02-21: added main function, added array export of data
#             added multiclass-SVM, added quick-PCA transformation, 
#             added accuracy check, added PCA-scatter visualisation, 
#             added extended reporting function, with histogram
# 2018-02-23: added PCA and KPCA as feature extraction methods,
#             reorganised so that run_classification takes all parameters
# 2018-02-26: used sklearn splitting, Kappa coefficients
# 2018-03-09:
import csv
import numpy as np
import pandas as pd


class SegmentCollection:
    '''Class for handling import of segments and classes as csv file
    segments_path = CSV file containing segments-IDs, features and target class
                    must contain columns like this:
                    system:index, class, .geo, feature1, feature2,... featureN
                    
    classes_path = CSV file containing class names and assigned colours
                    must contain columns like this:
                    class, label, colour
    '''
      
    def __init__(self,*, folder, segments_path, labels_path, classes_path):
        #import labelled segments
        labels = pd.read_csv(labels_path).drop(['ID'], axis=1)
        
        out = labels
        
        for path in segments_path:
            segments = pd.read_csv(folder + '/' + path).drop(['ID','.geo'], axis=1)
            out = pd.merge(out, segments, on='system:index')

        #import labelled segments
        self.labelled = out[out['class'] >=0]
        self.labelled_id = self.labelled['system:index']
        self.labelled_target = self.labelled['class']
        self.labelled_data = self.labelled.drop(['system:index','class','.geo'],axis=1)
        
        #import unlabelled segments
        self.unlabelled = out[out['class'] == -1]
        self.unlabelled_id = self.unlabelled['system:index']
        self.unlabelled_target = self.unlabelled['class']
        self.unlabelled_data = self.unlabelled.drop(['system:index','class','.geo'],axis=1)
        
        #import all segments
        self.all = out
        self.all_id = self.all['system:index']
        self.all_target = self.all['class']
        self.all_data = self.all.drop(['system:index','class','.geo'],axis=1)
        
        #list feature names
        self.feature_names = list(out.drop(['system:index', 'class', '.geo'], axis=1))
              
        #import class names
        self.classes = pd.read_csv(classes_path)
        
    def get_labelled(self):
        'returns idx, data and target as numpy arrays'
        return (self.labelled_id.values, self.labelled_data.values, self.labelled_target.values)
    
    def get_unlabelled(self):
        'returns idx, data and target as numpy arrays'
        return (self.unlabelled_id.values, self.unlabelled_data.values, self.unlabelled_target.values)
    
    def get_all(self):
        'returns idx, data and target as numpy arrays'
        return (self.all_id.values, self.all_data.values, self.all_target.values)
    
    def get_classes(self):
        'returns class, label and assigned colour'
        return(self.classes['class'].values, self.classes['label'].values, self.classes['colour'].values)
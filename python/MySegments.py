"""Module for handling Segments and Segmentcollection

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    author: Oliver Stromann
    created: 2018-02-13
    last updated: 2018-11-19
"""
import csv
import numpy as np
import pandas as pd

def mem_usage(pandas_obj):
     """Returns the memory usage of a Pandas DataFrame

    Args:
        pandas_obj (:obj:): Pandas DataFrame object.

    Returns:
        str: formatted memory usage of DataFrame.
    """
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
        
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


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
        """Import labelled segments

        Note:
            May need to run merge_CSVs.py cut_CSVs.py, cut_labels.py first.
            

        Args:
            folder (str): Folder path that contains the CSV files of ee_exports.
            segments_path(str): List of CSV files to read in folder.
            labels_path (str): Path to CSV files that contains segment IDs and target labels.
            classes_path (str, deprecated): Path to CSV files that contains class look up table.

        """
        out = pd.read_csv(labels_path)
     
        for path in segments_path:
            segments = pd.read_csv(folder + '/' + path, low_memory=True)
                        
            out = pd.merge(out, segments, on='FID', how='inner')
            print('read file: ', path, mem_usage(segments), ', total memory usage: ', mem_usage(out))
              
        #import labelled segments
        self.labelled = out[out['class'] >=0]
        self.labelled_id = self.labelled['FID']
        self.labelled_target = self.labelled['class']
        self.labelled_data = self.labelled.drop(['FID','class'],axis=1)
        
        #import unlabelled segments
        self.unlabelled = out[out['class'] == -1]
        self.unlabelled_id = self.unlabelled['FID']
        self.unlabelled_target = self.unlabelled['class']
        self.unlabelled_data = self.unlabelled.drop(['FID','class'],axis=1)
        
        #list feature names
        self.feature_names = list(out.drop(['FID', 'class'], axis=1))
                      
        #import all segments
        self.all = out
        self.all_id = self.all['FID']
        self.all_target = self.all['class']
        self.all_data = self.all.drop(['FID','class'],axis=1)
        
 
    def get_labelled(self):
        'returns idx, data and target as numpy arrays'
        return (self.labelled_id.values, self.labelled_data.values, self.labelled_target.values)
    
    def get_unlabelled(self):
        'returns idx, data and target as numpy arrays'
        return (self.unlabelled_id.values, self.unlabelled_data.values, self.unlabelled_target.values)
    
    def get_classes(self):
        'returns class, label and assigned colour'
        return(self.classes['class'].values, self.classes['label'].values, self.classes['colour'].values)

    def get_all(self):
        'returns idx, data and target as numpy arrays'
        return (self.all_id.values, self.all_data.values, self.all_target.values)

"""Merge multiple CSV files that have the same prefix separated by '-'.

This script merges all CSV files in PATH that share the same prefix,
but have a different postfix separated by '-'. Removing duplicate segments
based on their FID.


Example:
        Exported tables from GEE are named <table>-<tile>.csv and will be
        merged to <table>.csv
        $ python merge_files.py

Attributes:
    None.
Todo:
    * Pass PATH as argument (is it more convenient?)
"""

import glob
import random
import os
import pandas as pd

#GLOBALS
PATH="../exports"
   
def find_filesets(path="."):
    csv_files = {}
    for name in glob.glob("{}/*-*.csv".format(path)):
        key = os.path.splitext(os.path.basename(name))[0].split('-')[0]
        csv_files.setdefault(key, []).append(name)

    for key,filelist in csv_files.items(): 
        print(key, filelist)
        create_merged_csv(key, filelist)

def create_merged_csv(key, filelist):
    with open('{}.csv'.format(key), 'w+b') as outfile:
        df1 = pd.DataFrame()
        for filename in filelist:
            df2 = pd.read_csv(filename).drop(['system:index', 'ID', '.geo', 'class'], axis=1, errors='ignore')
            if df1.empty:
                df1 = df2
                
            else:
                #df1 = pd.merge(df1,df2, on='system:index', how='left')
                df1 = pd.concat([df1,df2])
                df1 = df1.drop_duplicates(subset='FID', keep='last')
        df1.to_csv('{}.csv'.format(key), index=False)

if __name__ == "__main__":
    find_filesets(PATH)

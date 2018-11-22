
# coding: utf-8

# In[1]:


import pandas as pd
from os import listdir
from os.path import isfile, join

mypath = '../ee_exports/aggregate'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]


labels_path = '../labels_2018-09-18_FID.csv'
folder = mypath
segments_path= onlyfiles

labels = pd.read_csv(labels_path)
        
labelled_idx = labels[labels['class'] >= 0]
unlabelled_idx = labels[labels['class'] == -1]
del(labels)

for path in segments_path:
    #trial
    mylist = []

    for chunk in  pd.read_csv(folder + '/' + path, chunksize=20000):              
        mylist.append(chunk)

    segments = pd.concat(mylist, axis= 0)
    del(mylist)

    out = pd.merge(labelled_idx, segments, on='FID', how='inner')
    del(segments)

    out.drop(['system:index', 'class'], axis=1).to_csv('../ee_exports/aggregate/reduced/'+ path, index=False)
    del(out)
    


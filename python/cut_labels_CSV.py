import pandas as pd
from os import listdir
from os.path import isfile, join

#GLOBAL PARAMETERS

PATH = 'demo/labels.csv'

df = pd.read_csv(PATH)
df.drop(['FID','.geo'], axis=1, errors='ignore').to_csv(PATH, index=False)
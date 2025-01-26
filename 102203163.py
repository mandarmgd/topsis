# TOPSIS Implementation in python 
import pandas as pd 
from pandas.api.types import is_numeric_dtype
from scipy.stats import rankdata
import numpy as np
import sys 

if(len(sys.argv) != 5):
    print('Incorrect number of arguments')
    exit(1)

# Reading the csv file created
try:
    data = pd.read_csv(sys.argv[1])
except: 
    print('file not found')
    exit(1)

if (data.shape[1] < 3):
    print('Three or more columns required')
    exit(1)

# Handling of non-numeric values
data = data.iloc[:, 1:]

for name in np.array(data.columns):
    if not is_numeric_dtype(data[name]):
        print('only numeric values allowed')
        exit(1)

# Normalization
data = data.values 
rows = data.shape[0]
columns = data.shape[1]
norm = []
for j in range(columns):
    rootsum = 0
    for i in range(rows):
        rootsum = rootsum + (data[i][j]) ** 2
    rootsum = np.sqrt(rootsum)
    
    data[:, j] = data[:, j] / rootsum


# Applying weights 
weights = sys.argv[2]
try:
    weights = np.array([int(item) for item in weights.split(',')])
except:
    print('invalid input')
    exit(1)
if(len(weights) != columns):
    print('number of weights should be valid and equal to number of columns, check if the separators are correct')
    exit(1)
for j in range(columns):
    data[:, j] = data[:, j] * weights[j]

# Calculate ideal best and worst values 
impacts = sys.argv[3].split(',') 
allowed = {'+', '-'}
if(not np.isin(impacts, list(allowed)).all()):
    print('invalid input')
    exit(1)
if(len(impacts) != columns):
    print('number of impacts should be equal to number of columns, check if the separators are correct')
    exit(1)

vjpos = []
vjneg = []

for j in range(columns):
    if (impacts[j] == "+"): 
        vjpos.append(max(data[:, j]))
        vjneg.append(min(data[:, j]))
    else:
        vjpos.append(min(data[:, j]))
        vjneg.append(max(data[:, j]))

vjpos = np.float64(vjpos)
vjneg = np.float64(vjneg)

# Euclidean distance from ideal best and worst values 
spos = []
sneg = []
for i in range(rows): 
    distpos = 0
    distneg = 0
    for j in range(columns):
        distpos = distpos + (data[i][j] - vjpos[j]) ** 2
        distneg = distneg + (data[i][j] - vjneg[j]) ** 2
    distpos = np.sqrt(distpos)
    distneg = np.sqrt(distneg)
    spos.append(distpos)
    sneg.append(distneg)

spos = np.float64(spos)
sneg = np.float64(sneg)

# Assigning performance scores 
scores = []
for i in range(rows): 
    p = sneg[i] / (spos[i] + sneg[i])
    p = round(p, 3)
    scores.append(p)

scores = np.float64(scores)

# Assigning ranks based on scores  
ranks = len(scores) - rankdata(scores).astype(int) + 1

# Updating and saving the csv file with scores and ranks 
data = pd.read_csv('102203163-data.csv') 
data['Topsis Score'] = scores 
data['Rank'] = ranks 
data.to_csv(sys.argv[4], index = False)
























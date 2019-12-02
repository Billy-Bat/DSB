import pandas as pd
import numpy as np

#0# scale down the train data set
bulk = pd.read_csv('data/train.csv')
train_small = bulk.head(500000)
train_small.to_csv('data/train_small.csv', index=False)

to_eval = train_small[((train_small['event_code'] == 4100) | (train_small['event_code'] == 4110)) & (train_small['type'] == 'Assessment')]
gsession = to_eval['game_session']

#0.1# load the train data
train_small = pd.read_csv('data/train_small.csv')

#0.2# size down the train label data
target = pd.read_csv('data/train_labels.csv')
shared = []
for gs in gsession :
    row_id = target[target['game_session'] == gs].index
    shared.append(row_id.values[0])

target_small = target.ix[shared]
target_small.to_csv('data/target_small.csv', index=False)

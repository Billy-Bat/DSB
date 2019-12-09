import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


start_event = ('abc5811c','4b5efe37','65abac75','0086365d','87d743c1','fd20ea40','f93fc684','cc5087a3','5be391b5','7040c096')

train = pd.read_csv('data/train.csv', chunksize=10000)
train = next(train)


Insta_Id = train.groupby('installation_id')

for Id, Insta_Id in Insta_Id :
    for gs_id, gs_info in Insta_Id.groupby('game_session') :
        for row in gs_info['event_data'] :
            print(row)

import pandas as pd
import numpy as np


('event_id', 'game_session', 'timestamp', 'event_data',
'installation_id', 'event_count', 'event_code', 'game_time', 'title',
'type', 'world')



train = pd.read_csv('data/train.csv')

grouped_train = train.groupby('installation_id')

ids_todrop = pd.Index([])
for (id, installation_id) in grouped_train :
    if 'Assessment' not in installation_id.index :
        ids_todrop = ids_todrop.union(installation_id.index)

train_filter = train.drop(ids_todrop)
train_filter.to_csv('data/train_filtered.csv')

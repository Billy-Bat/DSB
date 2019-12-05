import pandas as pd
import numpy as np





test = pd.read_csv('data/test.csv', chunksize=10000)
test = next(test)



test_byInstaID = test.groupby('installation_id')


for i, (instaID, played) in enumerate(test_byInstaID) :
    # print(played['type'].value_counts().Assessment)
    for gs_id, gs_info in played.groupby('game_session') :
        # print(played['type'].value_counts())
        if gs_info['type'].iloc[0] == 'Assessment' :
            print(gs_info['event_id'].unique())


    if i > 4 :
        break

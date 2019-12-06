import pandas as pd
import numpy as np





test = pd.read_csv('data/test.csv', chunksize=100000)
test = next(test)


test_byInstaID = test.groupby('installation_id')


for i, (instaID, played) in enumerate(test_byInstaID) :
    print(played['type'].values[-2])
    if played['type'].values[-2] == 'Assessment' :
        print(played['game_session'].values[-1])
        print(played['game_session'].values[-2])
        print(played['game_session'].values[-3])
        print(played['game_session'].values[-4])
        print(played['game_session'].values[-17])
    # for gs_id, gs_info in played.groupby('game_session') :


    # if i > 4 :
    #     break

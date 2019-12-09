import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib.pylab as mpl
import matplotlib
import seaborn as sns
from sklearn.preprocessing import normalize
import matplotlib.cm as cm

wincode = {'Mushroom Sorter (Assessment)':4100, 'Bird Measurer (Assessment)':4110, 'Cart Balancer (Assessment)':4100, 'Cauldron Filler (Assessment)':4100, 'Chest Sorter (Assessment)':4100}

train = pd.read_csv('data/train.csv', chunksize=500000)
train = next(train)
label = pd.read_csv('data/train_labels.csv')
specs = pd.read_csv('data/specs.csv')

# Extract the labels that contains coordinates information
labels = specs[specs["args"].str.contains('the game screen coordinates of the mouse click', na=False, regex=True)]
labels_collection = labels['event_id'].unique()

# split the df by insta_Id
train_InstaID = train.groupby('installation_id')

# Create a list to store the clicked heatmap
ActivitiesTracked = ['Scrub-A-Dub', 'Chow Time', 'Sandcastle Builder (Activity)']
ClickMap = dict.fromkeys(ActivitiesTracked, {0:[], 1:[], 2:[], 3:[]})

for i, (insta_Id, played) in enumerate(train_InstaID) :
    x_collec, y_collec = dict.fromkeys(ActivitiesTracked, []), dict.fromkeys(ActivitiesTracked, [])
    for j, (gs_Id, gs_info) in enumerate(played.groupby('game_session')) :
        gs_type = gs_info['type'].iloc[0]
        gs_title = gs_info['title'].iloc[0]
        game_session = gs_info['game_session'].iloc[0]

        if gs_type == 'Assessment' :
            # Collect Current Assessment Data for future entries
            all_attempts = gs_info.query('event_code == {}'.format(wincode[gs_title]))
            # then, check the numbers of wins and the number of losses
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0
            if accuracy == 0: Group = 0
            elif accuracy == 1: Group = 3
            elif accuracy == 0.5: Group = 2
            else: Group = 1

            if len(list(x_collec.values())[0]) > 50 :
                for act in ActivitiesTracked :
                    ClickMap[act][Group].append([x_collec[act], y_collec[act]])

        for act in ActivitiesTracked :
            if gs_title == act :
                coordinates_events = gs_info['event_data'][gs_info['event_id'].isin(labels_collection)]
                if not coordinates_events.empty :
                    gs_x_collec, gs_y_collec = [], []
                    for eve in coordinates_events :
                        raw_data = json.loads(eve)
                        gs_x_collec.append(float(raw_data['coordinates']['x']/raw_data['coordinates']['stage_width']))
                        gs_y_collec.append(float(raw_data['coordinates']['y']/raw_data['coordinates']['stage_height']))
                    x_collec[act] += gs_x_collec
                    y_collec[act] += gs_y_collec


# General HeatMap
all_x = dict.fromkeys(ActivitiesTracked, {0:[], 1:[], 2:[], 3:[]})
all_y = dict.fromkeys(ActivitiesTracked, {0:[], 1:[], 2:[], 3:[]})
for act in ClickMap.keys() :
    for i in range(4) :
        for pointcollection in ClickMap[act][i] :
            all_x[act][i] += pointcollection[0]
            all_y[act][i] += pointcollection[1]

for act in ClickMap.keys() :
    for i in range(4) :
        fig, ax = plt.subplots()
        counts, xedges, yedges, im = plt.hist2d(all_x[act][i],all_y[act][i], bins=[np.arange(0,1, 0.005),np.arange(0,1, 0.005)],
                                    norm=matplotlib.colors.Normalize(), density=True, cmap='RdBu_r')
        # counts = normalize(counts)
        # plt.imshow(counts, interpolation='nearest', cmap=cm.gist_rainbow)

        ax.set_title('Activity {}\n Group {}'.format(act, i))
        fig.savefig('vizu/ClickMap/Group{}.png'.format(i))
        plt.show()



# plt.show()

# Split the train df by Accuracy Group using the train_label set

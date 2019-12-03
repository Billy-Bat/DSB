import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm

('event_id', 'game_session', 'timestamp', 'event_data',
'installation_id', 'event_count', 'event_code', 'game_time', 'title',
'type', 'world')



train = pd.read_csv('data/train.csv', chunksize=100000)
# total_r = train.shape[0]

#------------------------------------------------------------------------------#
# print('counting types distrib')
# uni_types = train['type'].value_counts(normalize=True, dropna=False)
# print(uni_types)
"""
Game          0.511002
Activity      0.391210
Assessment    0.081593
Clip          0.016196
pd.Series(data=(0.511002, 0.391210, 0.081593, 0.016196), index=('Game', 'Activity', 'Assessment', 'Clip'))
"""
# type_distrib = pd.Series(data=(0.511002, 0.391210, 0.081593, 0.016196), index=('Game', 'Activity', 'Assessment', 'Clip'))
# print(type_distrib)
# ax = sns.barplot(x=type_distrib.index, y=type_distrib.values)
# ax.set_title('Type Proportion')
# ax.set_xlabel('TYPE')
# ax.set_ylabel('Proportion')
# fig = plt.gcf()
# fig.savefig('vizu/typeProportion.png')
# plt.show()

# print('counting world distrib')
# world_types = train['world'].value_counts(normalize=True, dropna=False)
# print(world_types)
"""
MAGMAPEAK       0.442965
CRYSTALCAVES    0.285031
TREETOPCITY     0.269925
NONE            0.002079
"""
# ax = sns.barplot(x=world_types.index, y=world_types.values)
# ax.set_title('World Proportion')
# ax.set_xlabel('World')
# ax.set_ylabel('Proportion')
# fig = plt.gcf()
# fig.savefig('vizu/WorldProportion.png')
# plt.show()



#------------------------------------------------------------------------------#

#0.1# get type by world
# ordered_world = [] # this is the order of the panda groupby
# world_id = [0, 1, 2, 3]
# by_world = train.groupby('world')
# world_total_r = []
# proportions_list = []
# for (id, world) in by_world :
#     ordered_world.append(id)
#     proportions = world['type'].value_counts(normalize=True, dropna=False)
#     proportions_list.append(proportions)
#     world_total_r.append(world.shape[0])
#
# print(ordered_world)
#
# type_bar = [[], [], [], []]
# for j, tpe in enumerate(['Game', 'Activity', 'Assessment', 'Clip']) :
#     for i, wrld in enumerate(ordered_world) :
#         if tpe in proportions_list[i].index :
#             type_bar[j].append(float(proportions_list[i][tpe]*100))
#         else :
#             type_bar[j].append(0.0)
#
# print(type_bar)
# bottom3 = [j+i for i,j in zip(type_bar[0], type_bar[1])]
# bottom4 = [j+i for i,j in zip(type_bar[2], bottom3)]
#
# plt.bar(world_id, type_bar[0], color='#b361b4', width=0.8, label='Game')
# plt.bar(world_id, type_bar[1], bottom=type_bar[0], color='#ec6767', edgecolor='white', label='Activity')
# plt.bar(world_id, type_bar[2], bottom=bottom3, color='#f4c06f', edgecolor='white', label='Assessment')
# plt.bar(world_id, type_bar[3], bottom=bottom4, color='#78d0aa', edgecolor='white', label='Clip')
# plt.xticks(world_id, ordered_world)
# plt.xlabel('group')
# colors = ['#b361b4', '#ec6767', '#f4c06f', '#78d0aa']
# handles = [plt.Rectangle((0,0),1,1, color=clr) for clr in colors]
# plt.legend(handles, ['Game', 'Activity', 'Assessment', 'Clip'])
# ax = plt.gca()
# ax.set_title('Activity type by World')
#
# fig = plt.gcf()
# fig.savefig('vizu/WorldProportion.png')
# plt.show()


#-----------------------------------------------------------------------------#
train = next(train)
grouped_assessment = train[train['type'] == 'Assessment']
grouped_assessment = grouped_assessment.groupby('game_session')


for i, (gsession_tag, info) in enumerate(grouped_assessment) :
    attempts_true = 0
    attempts_false = 0
    if len(info.index) <= 1 :
        pass
    if ((4100 in info['event_code'].values) | (4110 in info['event_code'].values)):
        event_data = [json.loads(x) for x in info[info['event_code'] == 4100]['event_data'].values]
        for row in event_data :
            if row['correct'] == True :
                attempts_true += 1
            else :
                attempts_false += 1
        print(attempts_true)
        print(attempts_false)
    # if i == 100 :
    #     break
        # attempts_true =+ sum([nbr_ture[] for nbr_ture i]

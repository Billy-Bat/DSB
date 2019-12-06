import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import json
from tqdm import tqdm
from statistics import mean
from datetime import datetime
from operator import add

matplotlib.rcParams.update({'font.size': 8})

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
# start_ID = ('90d848e0', '3bfd1a65', 'f56e0afc')
# exit_ID = ('3393b68b', 'c7128948', '2b058fe3')
# submit_ID = ('25fa8af4', 'd122731b') # doesnt mean sucessfule
# train_instaID = train.groupby('installation_id')
#
# gs_perInsta = {}
# outliers = 0
# for i in range(800) :
#     gs_perInsta[i] = 0
# for i, (instaId, all_sess) in enumerate(train_instaID) :
#     grouped_assessment = all_sess.groupby('game_session')
#     total_gs_played = grouped_assessment.size().shape[0]
#     if total_gs_played in gs_perInsta :
#         gs_perInsta[total_gs_played] += 1
#     else :
#         outliers += 1
#
# ax = sns.barplot(x=list(gs_perInsta.keys()), y=list(gs_perInsta.values()), palette=sns.color_palette("GnBu_d"))
# ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
# ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
# ax.set_title('Average Number of Game Session \n outliers : {}'.format(outliers))
# ax.set_xlabel('Total Game Session')
# ax.set_ylabel('Number of Installation ID')
# plt.xticks(rotation='vertical')
# fig = plt.gcf()
# fig.savefig('vizu/GameSessionPlayed.png')
# plt.show()
#-----------------------------------------------------------------------------#
# train = next(train)
# count = train['title'].value_counts()
# matplotlib.rcParams.update({'font.size': 5})
# ax = sns.barplot(x=count.values, y=count.index, palette=sns.color_palette("BuGn_r", n_colors=len(count.index)))
# ax.set_xlabel('number of activites')
# ax.set_ylabel('activity title')
# ax.set_title('Activites Count')
#
#
# fig = plt.gcf()
# fig.savefig('vizu/ActivitiesCount')
# plt.show()

#-----------------------------------------------------------------------------#
# train = next(train)
# train['timestamp'] = pd.to_datetime(train['timestamp'])
# train_instaID = train.groupby('installation_id')
#
# weekdayCount = {0: 0, 1: 0, 2 : 0, 3 : 0,
#                 4 : 0, 5 : 0, 6 : 0}
# weekdayNames = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
# MonthCount = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,
#               7: 0, 8: 0, 9: 0, 10: 0, 11:0, 12:0}
# monthNames = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
#               'August', 'September', 'October', 'November', 'December']
# hourCount = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,
#               7: 0, 8: 0, 9: 0, 10: 0, 11:0, 12:0, 13:0,
#               14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0,
#               22:0, 23:0}
#
# for i, (instaID, played) in enumerate(train_instaID) :
#     for j, (gsID, info) in enumerate(played.groupby('game_session')):
#         raw_date = info['timestamp'].values[0] # start (ends before midnight ?, see [-1])
#         ts = (raw_date - np.datetime64('1970-01-01T00:00:00Z'))/np.timedelta64(1, 's')
#         date_ = datetime.utcfromtimestamp(ts)
#         weekday = (raw_date.astype('datetime64[D]').view('int64') - 4) % 7
#         month = raw_date.astype('datetime64[M]').astype(int) % 12 + 1
#         hour = date_.hour
#
#         hourCount[hour] += 1
#         MonthCount[month] += 1
#         weekdayCount[weekday] += 1
#
# ax = sns.barplot(x=list(hourCount.keys()), y=list(hourCount.values()))
# ax.set_xlabel('Hour of the Day')
# ax.set_ylabel('Number of Game Session')
# ax.set_title('Game Session per hour of the week')
#
# fig = plt.gcf()
# fig.savefig('vizu/GameSessionPerHour')
# plt.show()
#
# ax = sns.barplot(x=monthNames, y=list(MonthCount.values()))
# ax.set_xlabel('Month of the Year')
# ax.set_ylabel('Number of Game Session')
# ax.set_title('Game Session per month of the year')
# plt.xticks(rotation=45)
#
# fig = plt.gcf()
# fig.savefig('vizu/GameSessionPerMonth')
# plt.show()
#
# ax = sns.barplot(x=weekdayNames, y=list(weekdayCount.values()))
# ax.set_xlabel('Day of the Week')
# ax.set_ylabel('Number of Game Session')
# ax.set_title('Game Session per day of the Week')
# plt.xticks(rotation=45)
#
# fig = plt.gcf()
# fig.savefig('vizu/GameSessionPerDay')
# plt.show()

#-----------------------------------------------------------------------------#
# train = next(train)
# train_instaID = train.groupby('installation_id')
#
# Assessment = {'Bird Measurer (Assessment)' : {0:0, 1:0, 2:0, 3:0},
#               'Cart Balancer (Assessment)' : {0:0, 1:0, 2:0, 3:0},
#               'Cauldron Filler (Assessment)' : {0:0, 1:0, 2:0, 3:0},
#               'Chest Sorter (Assessment)' : {0:0, 1:0, 2:0, 3:0},
#               'Mushroom Sorter (Assessment)' : {0:0, 1:0, 2:0, 3:0}}
# for i, (instaID, played) in enumerate(train_instaID):
#     for j, (gsID, info) in enumerate(played.groupby('game_session')) :
#         if (info['type'].values[0] == 'Assessment') & (len(info.index) > 1) :
#             indexes_submission = info.index[(info['event_code'] == 4100) |(info['event_code'] == 4110)]
#             currentType = info['title'].iloc[0]
#             correct_attempts = 0
#             incorrect_attemps = 0
#
#             for id in indexes_submission :
#                 data = json.loads(info['event_data'][info.index == id].values[0])
#                 if data['correct'] == True :
#                     correct_attempts += 1
#                 elif data['correct'] == False : incorrect_attemps += 1
#
#             if (correct_attempts >= 1) & (incorrect_attemps == 0) : Assessment[currentType][3] += 1
#             elif (correct_attempts == 1) & (incorrect_attemps == 1) : Assessment[currentType][2] += 1
#             elif (correct_attempts == 1) & (incorrect_attemps > 2) : Assessment[currentType][1] += 1
#             elif (correct_attempts == 0) : Assessment[currentType][0] += 1
#
#
# zero = [x[0]/sum(x.values()) for x in Assessment.values()]
# wan = [x[1]/sum(x.values()) for x in Assessment.values()]
# toos = [x[2]/sum(x.values()) for x in Assessment.values()]
# tres = [x[3]/sum(x.values()) for x in Assessment.values()]
#
# ax = sns.barplot(x=list(Assessment.keys()), y=zero, color='#b361b4')
# sns.barplot(x=list(Assessment.keys()), y=wan, color='#ec6767', bottom=zero, ax=ax, alpha=0.6)
# sns.barplot(x=list(Assessment.keys()), y=toos, color='#f4c06f', bottom=list(map(add, zero, wan)), ax=ax, alpha=0.6)
# sns.barplot(x=list(Assessment.keys()), y=tres, color='#78d0aa', bottom=list(map(add, zero, map(add, wan, toos))), alpha=0.6)
# for i in range(len(zero)) :
#     ax.text(i, zero[i], round(zero[i], 2), horizontalalignment='center', fontdict={'size':4})
#     ax.text(i, wan[i]+zero[i], round(wan[i], 2), horizontalalignment='center', fontdict={'size':4})
#     ax.text(i, wan[i]+zero[i]+toos[i], round(toos[i], 2), horizontalalignment='center', fontdict={'size':4})
#     ax.text(i, wan[i]+zero[i]+toos[i]+tres[i], round(tres[i], 2), horizontalalignment='center', fontdict={'size':4})
#
# colors = ['#b361b4', '#ec6767', '#f4c06f', '#78d0aa']
# handles = [plt.Rectangle((0,0),1,1, color=clr) for clr in colors]
# plt.legend(handles, ['0', '1', '2', '3'])
# ax.set_title('Accuracy per Assessment')
#
# fig = plt.gcf()
# fig.savefig('vizu/AccuracyPerAssess.png')
# plt.show()
#-----------------------------------------------------------------------------#
# train = next(train)
# train_instaID = train.groupby('installation_id')
# accuracy_group = {0:[], 1:[], 2:[], 3:[]}
#
# for i, (instaID, played) in enumerate(train_instaID):
#     correct_attempts = 0
#     incorrect_attemps = 0
#     for j, (gs_id, gs_info) in enumerate(played.groupby('game_session')):
#         if gs_info['type'].iloc[0] == 'Assessment' :
#             start = pd.to_datetime(gs_info['timestamp'].iloc[0])
#             end = pd.to_datetime(gs_info['timestamp'].iloc[-1])
#             duration = (start - end).seconds
#
#             indexes_submission = gs_info.index[(gs_info['event_code'] == 4100) |(gs_info['event_code'] == 4110)]
#             for id in indexes_submission :
#                 data = json.loads(gs_info['event_data'][gs_info.index == id].values[0])
#                 if data['correct'] == True :
#                     correct_attempts += 1
#                 elif data['correct'] == False : incorrect_attemps += 1
#
#             if (correct_attempts >= 1) & (incorrect_attemps == 0) : accuracy_group[3].append(duration)
#             elif (correct_attempts == 1) & (incorrect_attemps == 1) : accuracy_group[2].append(duration)
#             elif (correct_attempts == 1) & (incorrect_attemps > 2) : accuracy_group[1].append(duration)
#             elif (correct_attempts == 0) : accuracy_group[0].append(duration)
#
#
# ax = sns.boxplot(data=list(accuracy_group.values()))
# ax.set_xlabel('Accuracy Group')
# ax.set_ylabel('Time (in sec)')
# ax.set_title('Average Time spent Assessment per Accuracy Group \n Outliers Cutoff Range (85000, 87000)')
# plt.ylim(85000, 87000)
# fig = plt.gcf()
# fig.savefig('vizu/TimeSpentPerAssess.png')
#
#
# plt.show()

#-----------------------------------------------------------------------------#
# train = next(train)
# train_instaID = train.groupby('installation_id')
# accuracy_group = {0:[], 1:[], 2:[], 3:[]}
#
# for i, (instaID, played) in enumerate(train_instaID):
#     Total_played = 0
#     for j, (gs_id, gs_info) in enumerate(played.groupby('game_session')):
#         if gs_info['type'].iloc[0] == 'Assessment' :
#             correct_attempts = 0
#             incorrect_attemps = 0
#             indexes_submission = gs_info.index[(gs_info['event_code'] == 4100) |(gs_info['event_code'] == 4110)]
#             for id in indexes_submission :
#                 data = json.loads(gs_info['event_data'][gs_info.index == id].values[0])
#                 if data['correct'] == True :correct_attempts += 1
#                 elif data['correct'] == False : incorrect_attemps += 1
#
#             if (correct_attempts >= 1) & (incorrect_attemps == 0) : accuracy_group[3].append(Total_played)
#             elif (correct_attempts == 1) & (incorrect_attemps == 1) : accuracy_group[2].append(Total_played)
#             elif (correct_attempts == 1) & (incorrect_attemps > 2) : accuracy_group[1].append(Total_played)
#             elif (correct_attempts == 0) : accuracy_group[0].append(Total_played)
#
#         start_time = pd.to_datetime(gs_info['timestamp'].iloc[0])
#         end_time = pd.to_datetime(gs_info['timestamp'].iloc[-1])
#         Total_played += (end_time - start_time).seconds
#
# ax = sns.boxplot(data=list(accuracy_group.values()))
# ax.set_xlabel('Accuracy Group')
# ax.set_ylabel('Total Time (in sec)')
# ax.set_title('Total Time played before Assessment \n Outliers Cutoff Range (0, 60000)')
# plt.ylim(0, 60000)
# fig = plt.gcf()
# fig.savefig('vizu/TimeSpentBeforeAssess.png')
# #
# #
# plt.show()

#-----------------------------------------------------------------------------#
# train = next(train) # Usually the last Assessment of the train set has no correspondin
# labels = pd.read_csv('data/train_labels.csv')
# train_instaID = train.groupby('installation_id')
# accuracy_group = {0:[], 1:[], 2:[], 3:[]}
#
# for i, (instaID, played) in enumerate(train_instaID):
#     Total_played = 0
#     for j, (gs_id, gs_info) in enumerate(played.groupby('game_session')):
#         if gs_info['type'].iloc[0] == 'Assessment' :
#             result = labels['accuracy_group'][labels['game_session'] == gs_id].values
#             if len(result) != 0 :
#                 accuracy_group[result[0]].append(Total_played)
#
#         start_time = pd.to_datetime(gs_info['timestamp'].iloc[0])
#         end_time = pd.to_datetime(gs_info['timestamp'].iloc[-1])
#         Total_played += (end_time - start_time).seconds
#
# ax = sns.boxplot(data=list(accuracy_group.values()))
# ax.set_xlabel('Accuracy Group')
# ax.set_ylabel('Total Time (in sec)')
# ax.set_title('Total Time played before Assessment \n Outliers Cutoff Range (0, 60000)')
# fig = plt.gcf()
# fig.savefig('vizu/TimeSpentBeforeAssess.png')
#
# plt.show()

#-----------------------------------------------------------------------------#
# train = next(train) # Usually the last Assessment of the train set has no correspondin
# labels = pd.read_csv('data/train_labels.csv')
# train_instaID = train.groupby('installation_id')
# accuracy_group = {0:[], 1:[], 2:[], 3:[]}
#
# for i, (instaID, played) in enumerate(train_instaID):
#     Total_events = 0
#     for j, (gs_id, gs_info) in enumerate(played.groupby('game_session')):
#         if gs_info['type'].iloc[0] == 'Assessment' :
#             result = labels['accuracy_group'][labels['game_session'] == gs_id].values
#             if len(result) != 0 :
#                 accuracy_group[result[0]].append(Total_events)
#
#         Total_events += gs_info.shape[0]
#
# ax = sns.boxplot(data=list(accuracy_group.values()))
# ax.set_xlabel('Accuracy Group')
# ax.set_ylabel('Total Number of Events')
# ax.set_title('Events encountered before Assessment \n')
# fig = plt.gcf()
# fig.savefig('vizu/EventsBeforeAssess.png')
# plt.show()

#-----------------------------------------------------------------------------#
# train = next(train) # Usually the last Assessment of the train set has no correspondin
# labels = pd.read_csv('data/train_labels.csv')
# train_instaID = train.groupby('installation_id')
# accuracy_group = {0:[], 1:[], 2:[], 3:[]}
# accuracy_group_pertitle = {'Bird Measurer (Assessment)' : {0:[], 1:[], 2:[], 3:[]},
#               'Cart Balancer (Assessment)' :  {0:[], 1:[], 2:[], 3:[]},
#               'Cauldron Filler (Assessment)' :  {0:[], 1:[], 2:[], 3:[]},
#               'Chest Sorter (Assessment)' : {0:[], 1:[], 2:[], 3:[]},
#               'Mushroom Sorter (Assessment)' : {0:[], 1:[], 2:[], 3:[]}}
#
# for i, (instaID, played) in enumerate(train_instaID):
#     Total_Assessment = 0
#     Total_Assessment_type = {'Bird Measurer (Assessment)' : 0,
#                   'Cart Balancer (Assessment)' :  0,
#                   'Cauldron Filler (Assessment)' :  0,
#                   'Chest Sorter (Assessment)' : 0,
#                   'Mushroom Sorter (Assessment)': 0} # Recording what type of Assess was taken
#
#     for j, (gs_id, gs_info) in enumerate(played.groupby('game_session')):
#         if gs_info['type'].iloc[0] == 'Assessment' :
#             result = labels['accuracy_group'][labels['game_session'] == gs_id].values
#             title = labels['title'][labels['game_session'] == gs_id].values
#             if len(result) != 0 :
#                 accuracy_group[result[0]].append(Total_Assessment)
#                 accuracy_group_pertitle[title[0]][result[0]].append(Total_Assessment_type[title[0]])
#                 Total_Assessment += 1
#                 Total_Assessment_type[title[0]] += 1

# #1# plot of Assessment taken vs. accuracy_group
# ax = sns.boxplot(data=list(accuracy_group.values()))
# ax.set_xlabel('Accuracy Group')
# ax.set_ylabel('Total Number Assessment taken')
# ax.set_title('Number of Assessment Taken before \n')
# fig = plt.gcf()
# fig.savefig('vizu/AssessmentTaken.png')
# plt.show()

# #2# plot of Assessment taken (per type) vs. accuracy_group
# barWidth = 0.25
# typelist = ['Bird Measurer (Assessment)', 'Cart Balancer (Assessment)', 'Cauldron Filler (Assessment)', 'Chest Sorter (Assessment)', 'Mushroom Sorter (Assessment)']
# place_typ1 = np.arange(4)*2
# err_type1 = list(map(np.std, list(accuracy_group_pertitle['Bird Measurer (Assessment)'].values())))
# mean_type1 = list(map(np.mean, list(list(accuracy_group_pertitle['Bird Measurer (Assessment)'].values()))))
# place_typ2 = [x + barWidth for x in place_typ1]
# err_type2 = list(map(np.std, list(accuracy_group_pertitle['Cart Balancer (Assessment)'].values())))
# mean_type2 = list(map(np.mean, list(list(accuracy_group_pertitle['Cart Balancer (Assessment)'].values()))))
# place_typ3 = [x + barWidth for x in place_typ2]
# err_type3 = list(map(np.std, list(accuracy_group_pertitle['Cauldron Filler (Assessment)'].values())))
# mean_type3 = list(map(np.mean, list(list(accuracy_group_pertitle['Cauldron Filler (Assessment)'].values()))))
# place_typ4 = [x + barWidth for x in place_typ3]
# err_type4 = list(map(np.std, list(accuracy_group_pertitle['Chest Sorter (Assessment)'].values())))
# mean_type4 = list(map(np.mean, list(list(accuracy_group_pertitle['Chest Sorter (Assessment)'].values()))))
# place_typ5 = [x + barWidth for x in place_typ4]
# err_type5 = list(map(np.std, list(accuracy_group_pertitle['Mushroom Sorter (Assessment)'].values())))
# mean_type5 = list(map(np.mean, list(list(accuracy_group_pertitle['Mushroom Sorter (Assessment)'].values()))))
#
#
# ax = plt.bar(place_typ1, mean_type1, yerr=err_type1, align='center', width=barWidth, color='#00cbb5')
# plt.bar(place_typ2, mean_type2, yerr=err_type2, align='center', width=barWidth, color='#fbc445')
# plt.bar(place_typ3, mean_type3, yerr=err_type3, align='center', width=barWidth, color='#070033')
# plt.bar(place_typ4, mean_type4, yerr=err_type4, align='center', width=barWidth, color='#bf1e2e')
# plt.bar(place_typ5, mean_type5, yerr=err_type5, align='center', width=barWidth, color='#bada55')
#
# colors = ['#00cbb5', '#fbc445', '#070033', '#bada55']
# handles = [plt.Rectangle((0,0),1,1, color=clr) for clr in colors]
# plt.legend(handles, typelist)
# plt.xticks(place_typ3, [0, 1, 2, 3])
# plt.gca().set_ylim(bottom=0)
# plt.gca().set_title('Assessment Taken (Per Type) per accuracy group')
# plt.gca().set_ylabel('Nbr of Assessment')
# plt.gcf().savefig('vizu/AssessmentTakenPerType.png')
#
# plt.show()
#-----------------------------------------------------------------------------#
# train = next(train) # Usually the last Assessment of the train set has no correspondin
# train['timestamp'] = pd.to_datetime(train['timestamp'])
# labels = pd.read_csv('data/train_labels.csv')
# accuracy_group_perday ={ 0 : {0:0, 1:0, 2:0, 3:0},
#                           1 : {0:0, 1:0, 2:0, 3:0},
#                           2 : {0:0, 1:0, 2:0, 3:0},
#                           3 : {0:0, 1:0, 2:0, 3:0},
#                           4 : {0:0, 1:0, 2:0, 3:0},
#                           5 : {0:0, 1:0, 2:0, 3:0},
#                           6 : {0:0, 1:0, 2:0, 3:0},}
#
# for i, (instaID, played) in enumerate(train.groupby('installation_id')):
#     Total_Assessment = 0
#     for j, (gs_id, gs_info) in enumerate(played.groupby('game_session')):
#         if gs_info['type'].iloc[0] == 'Assessment' :
#             result = labels['accuracy_group'][labels['game_session'] == gs_id].values
#             title = labels['title'][labels['game_session'] == gs_id].values
#             date = (gs_info['timestamp'].values[0].astype('datetime64[D]').view('int64') - 4) % 7
#             if len(result) != 0 :
#                 accuracy_group_perday[date][result[0]] += 1
#
# barWidth = 0.25
# acc0_placement = np.arange(7)*2
# acc1_placement = [x + barWidth for x in acc0_placement]
# acc2_placement = [x + barWidth for x in acc1_placement]
# acc3_placement = [x + barWidth for x in acc2_placement]
# plt.bar(acc0_placement, [accuracy_group_perday[x][0]/sum(accuracy_group_perday[x].values()) for x in accuracy_group_perday.keys()], align='center', width=barWidth, color='#fbc445')
# plt.bar(acc1_placement, [accuracy_group_perday[x][1]/sum(accuracy_group_perday[x].values()) for x in accuracy_group_perday.keys()], align='center', width=barWidth, color='#070033')
# plt.bar(acc2_placement, [accuracy_group_perday[x][2]/sum(accuracy_group_perday[x].values()) for x in accuracy_group_perday.keys()], align='center', width=barWidth, color='#bf1e2e')
# plt.bar(acc3_placement, [accuracy_group_perday[x][3]/sum(accuracy_group_perday[x].values()) for x in accuracy_group_perday.keys()], align='center', width=barWidth, color='#bada55')
#
# plt.xticks(acc1_placement, ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
# colors = ['#fbc445', '#070033', '#bf1e2e', '#bada55']
# handles = [plt.Rectangle((0,0),1,1, color=clr) for clr in colors]
# plt.legend(handles, [0, 1, 2, 3])
# plt.gca().set_title('Accuracy Proportion per Day of the Week')
# plt.gca().set_ylabel('Proportion')
# plt.gcf().savefig('vizu/AccuracyGroupPerDay.png')
#
# plt.show()
#-----------------------------------------------------------------------------#
# train = next(train) # Usually the last Assessment of the train set has no correspondin
# train['timestamp'] = pd.to_datetime(train['timestamp'])
# labels = pd.read_csv('data/train_labels.csv')
# accuracy_perhour = {}
# for i in range(24) :
#     accuracy_perhour[i] = {0:0, 1:0, 2:0, 3:0}
#
# def div_zero(A, B) :
#     return A/B if B else 0
#
# for i, (instaID, played) in enumerate(train.groupby('installation_id')):
#     Total_Assessment = 0
#     for j, (gs_id, gs_info) in enumerate(played.groupby('game_session')):
#         if gs_info['type'].iloc[0] == 'Assessment' :
#             result = labels['accuracy_group'][labels['game_session'] == gs_id].values
#             title = labels['title'][labels['game_session'] == gs_id].values
#             raw_date = gs_info['timestamp'].values[0]
#             ts = (raw_date - np.datetime64('1970-01-01T00:00:00Z'))/np.timedelta64(1, 's')
#             date_ = datetime.utcfromtimestamp(ts)
#             hour = date_.hour
#
#             date = (gs_info['timestamp'].values[0].astype('datetime64[D]').view('int64') - 4) % 7
#             if len(result) != 0 :
#                 accuracy_perhour[hour][result[0]] += 1
#
# barWidth = 0.25
# acc0_placement = np.arange(24)*2
# acc1_placement = [x + barWidth for x in acc0_placement]
# acc2_placement = [x + barWidth for x in acc1_placement]
# acc3_placement = [x + barWidth for x in acc2_placement]
# plt.bar(acc0_placement, [div_zero(accuracy_perhour[x][0], sum(accuracy_perhour[x].values())) for x in accuracy_perhour.keys()], align='center', width=barWidth, color='#fbc445')
# plt.bar(acc1_placement, [div_zero(accuracy_perhour[x][1], sum(accuracy_perhour[x].values())) for x in accuracy_perhour.keys()], align='center', width=barWidth, color='#070033')
# plt.bar(acc2_placement, [div_zero(accuracy_perhour[x][2], sum(accuracy_perhour[x].values())) for x in accuracy_perhour.keys()], align='center', width=barWidth, color='#bf1e2e')
# plt.bar(acc3_placement, [div_zero(accuracy_perhour[x][3], sum(accuracy_perhour[x].values())) for x in accuracy_perhour.keys()], align='center', width=barWidth, color='#bada55')
#
# plt.xticks(acc1_placement, list(accuracy_perhour.keys()))
# colors = ['#fbc445', '#070033', '#bf1e2e', '#bada55']
# handles = [plt.Rectangle((0,0),1,1, color=clr) for clr in colors]
# plt.legend(handles, [0, 1, 2, 3])
# plt.gca().set_title('Accuracy Proportion per Hour of the day')
# plt.gca().set_ylabel('Proportion')
# plt.gcf().savefig('vizu/AccuracyGroupPerHour.png')
# plt.show()

#-----------------------------------------------------------------------------#
# train = next(train) # Usually the last Assessment of the train set has no correspondin
# labels = pd.read_csv('data/train_labels.csv')
# accuracy_group_vs_correct = {0:[], 1:[], 2:[], 3:[]}
# accuracy_group_vs_incorrect = {0:[], 1:[], 2:[], 3:[]}
# acc_group = {0:[], 1:[], 2:[], 3:[]}
#
# for i, (instaID, played) in enumerate(train.groupby('installation_id')):
#     Total_Assessment = 0
#     accum_correct = 0
#     accum_incorrect = 0
#     accum_accurracy = 0
#     prev_accurracy = []
#     for j, (gs_id, gs_info) in enumerate(played.groupby('game_session')):
#         if gs_info['type'].iloc[0] == 'Assessment' :
#             result = labels['accuracy_group'][labels['game_session'] == gs_id].values
#             indexes_submission = gs_info.index[(gs_info['event_code'] == 4100) |(gs_info['event_code'] == 4110)]
#             # currentType = info['title'].iloc[0]
#             correct_attempts = 0
#             incorrect_attemps = 0
#             group_accurracy = 0
#
#
#             for id in indexes_submission :
#                 data = json.loads(gs_info['event_data'][gs_info.index == id].values[0])
#                 if data['correct'] == True : correct_attempts += 1
#                 elif data['correct'] == False : incorrect_attemps += 1
#
#
#             if (correct_attempts >= 1) & (incorrect_attemps == 0) :
#                 accuracy_group_vs_correct[3].append(accum_correct),  accuracy_group_vs_incorrect[3].append(accum_incorrect)
#                 acc_group[3].append(accum_accurracy)
#                 group_accurracy = 3
#             elif (correct_attempts == 1) & (incorrect_attemps == 1) :
#                 accuracy_group_vs_correct[2].append(accum_correct), accuracy_group_vs_incorrect[2].append(accum_incorrect)
#                 acc_group[2].append(accum_accurracy)
#                 group_accurracy = 2
#             elif (correct_attempts == 1) & (incorrect_attemps > 2) :
#                 accuracy_group_vs_correct[1].append(accum_correct), accuracy_group_vs_incorrect[1].append(accum_incorrect)
#                 acc_group[1].append(accum_accurracy)
#                 group_accurracy = 1
#             elif (correct_attempts == 0) :
#                 accuracy_group_vs_correct[0].append(accum_correct), accuracy_group_vs_incorrect[0].append(accum_incorrect)
#                 acc_group[0].append(accum_accurracy)
#                 group_accurracy = 0
#
#
#             accum_correct += correct_attempts
#             accum_incorrect += incorrect_attemps
#
#             prev_accurracy.append(group_accurracy)
#             acc_group[group_accurracy].append(mean(prev_accurracy))
#
#
#
# #1# Correct vs Acc Group
# ax = sns.barplot(x=[0, 1, 2, 3], y=list(map(np.mean, accuracy_group_vs_correct.values())), yerr=list(map(np.std, accuracy_group_vs_correct.values())))
# ax.set_xlabel('Accuracy Group')
# ax.set_ylabel('Accumulated Correct Attempt')
# plt.gca().set_title('Accumulated Correct Attempt vs Accuracy Group')
#
# plt.xticks(range(4), [0, 1, 2, 3])
# plt.gcf().savefig('vizu/AccumCorrectVSAccuracyGroup')
# plt.show()
#
# #2# InCorrect vs Acc Group
# ax = sns.barplot(x=[0, 1, 2, 3], y=list(map(np.mean, accuracy_group_vs_incorrect.values())), yerr=list(map(np.std, accuracy_group_vs_incorrect.values())))
# ax.set_xlabel('Accuracy Group')
# ax.set_ylabel('Accumulated Incorrect Attempt')
# plt.gca().set_title('Accumulated Incorrect Attempt vs Accuracy Group')
#
# plt.xticks(range(4), [0, 1, 2, 3])
# plt.gcf().savefig('vizu/AccumIncorrectVSAccuracyGroup')
# plt.show()
# #3# Accumulated Accuracy vs Accuracy Group
# ax = sns.barplot(x=[0, 1, 2, 3], y=list(map(np.mean, acc_group.values())), yerr=list(map(np.std, acc_group.values())))
# ax.set_xlabel('Accuracy Group')
# ax.set_ylabel('Accumulated Accuracy')
# plt.gca().set_title('Accumulated Accuracy Group vs Accuracy Group')
#
# plt.show()
#-----------------------------------------------------------------------------#
train = next(train) # Usually the last Assessment of the train set has no correspondin
[4070, 4030, 3010, 3110, 4020, 4035, 2020, 4021, 2030, 4025, 2000,
            3021, 3121, 3020, 3120, 4022, 4040, 4031, 2040, 2050, 2080, 4010,
            4230, 4045, 4235, 4090, 5000, 2083, 5010, 4100, 4220, 4110, 2025,
            2060, 2035, 2081, 4080, 2070, 2010, 4095, 2075]

uniquestuff = {}
for i, (instaID, played) in enumerate(train.groupby('installation_id')):

    for j, (gs_id, gs_info) in enumerate(played.groupby('game_session')):
        interest = gs_info[gs_info['event_id'] == '67aa2ada']
        if not interest.empty :
            print(interest['event_code'].unique())

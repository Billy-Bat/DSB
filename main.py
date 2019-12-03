import pandas as pd
import numpy as np
import json
from tqdm import tqdm

cols_to_drop = ['game_session', 'installation_id', 'timestamp', 'accuracy_group', 'timestampDate']

""" INDEX
('event_id', 'game_session', 'timestamp', 'event_data',
'installation_id', 'event_count', 'event_code', 'game_time', 'title',
'type', 'world')
    DTYPES
event_id           object
game_session       object
timestamp          object
event_data         object
installation_id    object
event_count         int64
event_code          int64
game_time           int64
title              object
type               object
world              object
dtype: object

    TARGET INDEX
('game_session', 'installation_id', 'title', 'num_correct',
       'num_incorrect', 'accuracy', 'accuracy_group')
   DTYPES
game_session        object
installation_id     object
title               object
num_correct          int64
num_incorrect        int64
accuracy           float64
accuracy_group       int64

TOTAL USERS : 1000

EventCode : 4100 4110
"""


def get_installationID_data(userdf) :
    last_activity = 0
    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
    last_session_time_sec = 0
    accuracy_groups = {0:0, 1:0, 2:0, 3:0}
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy = 0
    accumulated_correct_attempts = 0
    accumulated_uncorrect_attempts = 0
    accumulated_actions = 0
    counter = 0

    last_accuracy_title = {'acc_' + title : -1 for title in assess_titles} # dic of all values

    features = user_activities_count.copy()
    features.update(last_accuracy_title.copy())

    for i, session in userdf.groupby('game_session', sort=False):
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        session_title_text = activities_labels[session_title]
        print(session_type)

    return 0

def encode_title(train, test, train_labels):
    # encode title
    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))
    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))
    # make a list with all the unique 'titles' from the train and test set
    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))
    # make a list with all the unique 'event_code' from the train and test set
    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))
    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))
    # make a list with all the unique worlds from the train and test set
    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))
    # create a dictionary numerating the titles
    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))
    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))
    # replace the text titles with the number titles from the dict
    train['title'] = train['title'].map(activities_map)
    test['title'] = test['title'].map(activities_map)
    train['world'] = train['world'].map(activities_world)
    test['world'] = test['world'].map(activities_world)
    train_labels['title'] = train_labels['title'].map(activities_map)
    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))
    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
    win_code[activities_map['Bird Measurer (Assessment)']] = 4110
    # convert text into datetime
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])

    return train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code


if __name__ == '__main__' :

    #1# load the train data
    train = pd.read_csv('data/train_small.csv')

    #2# load the training data
    target = pd.read_csv('data/train_small.csv')

    #2# load the test_data
    test = pd.read_csv('data/test.csv')

    train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels,
    assess_titles, list_of_event_id, all_title_event_code = encode_title(train, test, train)

    # Group Users
    # for i, (id, insta_id) in enumerate(train.groupby('installation_id')) :
    #     # Group game sessions
    #     for j, (id, gsession) in enumerate(train.groupby('game_session')) :
    #
    #         session_type = gsession['type'].iloc[0]
    #         session_title = gsession['title'].iloc[0]
    #         session_title_text = activities_labels[session_title]
    #
    #         print(gsession)



    # encode title
    # train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
    # train['timestamp'] = pd.to_datetime(train['timestamp'])
    #
    # title_eventlist = list(set(train["title_event_code"].unique()))
    # listuser_act = list(set(train['title'].unique()))
    # listeven_code = list(set(train['event_code'].unique()))
    # listevent_id = list(set(train['event_id'].unique()))
    # listworlds = list(set(train['world'].unique()))
    # assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index))
    # activities_labels = dict(zip(np.arange(len(listuser_act)), listuser_act))
    #
    # for i, (gs_id, user_data) in enumerate(train.groupby('installation_id')) :
    #     get_installationID_data(user_data)
    #
    #
    #
    # def format_data(trainData) :
    #     return reduced_train

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

def div_zero(A, B) :
    return A/B if B else 0

def get_insta_Data(instal_Data, test=False) :
    """

    """
    Assessments = []
    wincode = {'Mushroom Sorter (Assessment)':4100, 'Bird Measurer (Assessment)':4110, 'Cart Balancer (Assessment)':4100, 'Cauldron Filler (Assessment)':4100, 'Chest Sorter (Assessment)':4100}
    cols = {'game_session':0, 'session_title':0, 'hour':0, 'total_spent':0, 0:0, 1:0, 2:0, 3:0, 'accum_accuracy':0, 'accum_accuracy_group':0,
            'accum_correct':0, 'accum_incorrect':0, 'last_accuracyGroup': 0, 'Total_events':0, 'Total_Assessment':0, 'helped':0, 'Target':0}
    Activity_count = {'Clip':0, 'Game':0, 'Activity':0, 'Assessment':0}
    AssessmentsTaken = {'Mushroom Sorter (Assessment)':0, 'Bird Measurer (Assessment)':0, 'Cart Balancer (Assessment)':0, 'Cauldron Filler (Assessment)':0, 'Chest Sorter (Assessment)':0}
    AccGroupCount = {0:0, 1:0, 2:0, 3:0}

    # number and accumulated corr/incorr attemps + Accum. and Accuracy Group Count
    accum_correct, accum_incorrect = 0, 0
    accum_accuracy, accum_accuracy_group = 0, 0
    # last_accuracyGroup & Total Assessment Taken
    last_accuracyGroup = -1
    Total_Assessment = 0
    # Last Session Type
    last_session = 0
    # Total time spent playing
    total_spent = 0
    # accumulated Actions
    accumulated_actions = 0
    # total events encountered
    total_event = 0
    # total help
    helped = 0

    # time spent on each
    # time_spent_each_act = {actv: 0 for actv in list_of_user_activities}

    for j, (gs_id, gs_info) in enumerate(instal_Data.groupby('game_session')):
        gs_type = gs_info['type'].iloc[0]
        gs_title = gs_info['title'].iloc[0]
        game_session = gs_info['game_session'].iloc[0]

        if gs_type == 'Assessment' :
            #1# Append all the data collected so far (excluding current Assessment)
            # Create the entry dictionary
            features = cols.copy()
            features.update(Activity_count)
            features.update(AssessmentsTaken)
            # Update the dictionary with value extracted from the current session
            features['game_session'] = game_session
            features['session_title'] = gs_title
            features['hour'] = pd.Timestamp(gs_info['timestamp'].iloc[0]).hour
            features['total_spent'] = total_spent
            for acc_group in AccGroupCount.keys() :
                features[acc_group] = AccGroupCount[acc_group]
            features['accum_accuracy'] = accum_accuracy
            # features['accum_accuracy_group'] =
            features['accum_correct'] = accum_correct
            features['accum_incorrect'] = accum_incorrect
            features['last_accuracyGroup'] = last_accuracyGroup
            features['Total_Assessment'] = Total_Assessment
            features['total_event'] = total_event
            features['helped'] = helped
            for act in Activity_count :
                features[act] += Activity_count[act]
            #### APEND IS AT THE END FOR THE TARGET VALUE


            #2# Collect Current Assessment Data for future entries
            all_attempts = gs_info.query('event_code == {}'.format(wincode[gs_title]))
            # then, check the numbers of wins and the number of losses
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0
            accum_accuracy += accuracy
            # update the accumulated count of correct/incorr guesses
            accum_correct += true_attempts
            accum_incorrect += false_attempts

            if accuracy == 0:
                last_accuracyGroup = 0
            elif accuracy == 1:
                last_accuracyGroup = 3
            elif accuracy == 0.5:
                last_accuracyGroup = 2
            else:
                last_accuracyGroup = 1

            # For the next Assessment we save the count of accuracies groups
            AccGroupCount[last_accuracyGroup] += 1 if last_accuracyGroup != -1 else 0
            # The Accumulated accuracy update
            accum_accuracy += last_accuracyGroup if last_accuracyGroup != 1 else 0
            # The group Accumulated accuracy update # BASICALLY ENCODE SAME INFO AS ABOVE
            accum_accuracy_group = sum(AccGroupCount.values())
            # Update total Assessment and count per Assessment
            Total_Assessment += 1
            AssessmentsTaken[gs_title] += 1

            #### APPEND THE FEATURE ####
            features['Target'] = last_accuracyGroup
            Assessments.append(features)

        #3# Collect Non Assessment Based Data

        event_info = gs_info['event_code'].value_counts()
        helped +=  event_info[4090] if 4090 in event_info.index else 0
        total_event += gs_info.shape[0]
        total_spent += (pd.to_datetime(gs_info.iloc[-1, 2]) - pd.to_datetime(gs_info.iloc[0, 2])).seconds
        Activity_count[gs_type] += 1



    if test :
        return Assessments[-1]
    else :
        return Assessments[0:-1]

if __name__ == '__main__' :
    dic_ = {'game_session':0, 'session_title':0, 'hour':0, 'total_spent':0, 0:0, 1:0, 2:0, 3:0, 'accum_accuracy':0, 'accum_accuracy_group':0,
            'accum_correct':0, 'accum_incorrect':0, 'last_accuracyGroup': 0}
    reduced_train = pd.DataFrame(columns=dic_.keys())
    # print(reduced_train)

    #1# load the train data
    train = pd.read_csv('data/train.csv', chunksize=100000)
    train = next(train)
    for insta_Id, insta_info, in train.groupby('installation_id') :
        Data = get_insta_Data(insta_info)
        if Data :
            for entry in Data :
                reduced_train = reduced_train.append(entry, ignore_index=True)

    # #2# load the training data
    # target = pd.read_csv('data/train_labels.csv')



    # #2# load the test_data
    # test = pd.read_csv('data/test.csv')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit, RepeatedStratifiedKFold

def corr_heatmap(Pred, Data, method='pearson', _Save=False):
    full = pd.concat((Pred, Data), axis=1, sort=True)

    res = full.corr(method=method)
    res = res.drop(['Total_events'])
    print(res)
    pc = sns.heatmap(res, vmin=-1, vmax=1, center=0, cmap='RdBu_r')

    axes = plt.gca()
    axes.set_title(method)
    axes.set_xticks(np.arange(res.shape[1]) + 0.5)
    axes.set_yticks(np.arange(res.shape[0]) + 0.5)
    axes.set_xticklabels(list(res.columns), rotation=90, fontdict={'weight': 'normal','size': 4})
    axes.set_yticklabels(list(res.index), fontdict={'weight': 'normal','size': 4})
    # plt.colorbar(pc, cax=axes)

    if _Save :
        fig = plt.gcf()
        fig.savefig('vizu/CorrelHeatMap.png')

    plt.show()

    return 0

def outlier_analysis(Pred, Data, include=None, trsh=0.99, _Save=False):
    indexes = []
    if include == None :
        iterable = Data.columns
    else :
        iterable = include

    for col in iterable :
        if (col != 'game_session') &  (col != 'session_title'):
            data = pd.concat((Pred, Data[col]), axis=1, sort=True)
            Pred_quant = Pred.quantile(trsh)
            Data_quant = Data[col].quantile(trsh)
            s1 = sns.jointplot(x=col, y=data.columns[0], data=data, kind='reg')
            s1.ax_joint.plot([0, Data[col].max()*1.1], [Pred_quant, Pred_quant], linewidth=2)
            s1.ax_joint.plot([Data_quant, Data_quant], [0, Pred.max()*1.1], linewidth=2)
            s1.set_axis_labels(xlabel=col, ylabel=data.columns[0])
            s1.annotate(stats.pearsonr)

            plt.subplots_adjust(top=0.9)
            s1.fig.suptitle('{} Outlier Analysis \n quantile at {}'.format(col, trsh))

            pear_r = stats.pearsonr(data.iloc[:,0], data.iloc[:,1])[0]
            if pear_r > 0.5 :
                id_above = Data.loc[Data[col] > Data_quant].index
                coresp = Pred.ix[id_above]
                id_outlier = coresp.loc[Pred.iloc[:,0] < Pred_quant].index
                print(id_outlier)

            if _Save :
                fig = plt.gcf()
                fig.savefig('reduced_vizu/outlier/{}_outlierAnalysis.png'.format(col))

        plt.show()

    return indexes


reduced = pd.read_csv('data/reduced_train.csv')
Target = reduced['Target']
reduced = reduced.drop(['Target'], axis=1)
reduced['total_spent'] = np.log1p(reduced['total_spent'])
reduced['accum_accuracy'] = np.log1p(reduced['accum_accuracy'])
reduced = pd.get_dummies(reduced, prefix=['Type'], columns=['session_title'])
# sns.distplot(reduced['total_spent'])
# plt.show()
# sns.distplot(reduced['accum_accuracy'])
# plt.show()


# corr_heatmap(Target, reduced)
# outlier_analysis(Target, reduced, _Save=True)

param = {'n_estimators':2000,
         'learning_rate': 0.01,
         'metric': 'multiclass',
         'objective': 'multiclass',
         'max_depth': 15,
         'num_classes': 4,
         'feature_fraction': 0.85,
         'subsample': 0.85,
         'verbose': 1000,
         'early_stopping_rounds': 100, 'eval_metric': 'cappa'
        }

y = data['accuracy_group']
n_fold = 5
#folds = StratifiedKFold(n_splits=n_fold)
#folds = KFold(n_splits=n_fold)
folds = RepeatedStratifiedKFold(n_splits=n_fold)
folds = GroupKFold(n_splits=n_fold)
cols_to_drop = ['game_session']
cat_cols = ['world']
mt = MainTransformer(create_interactions=False)
ct = CategoricalTransformer(drop_original=True, cat_cols=cat_cols)
ft = FeatureTransformer()
transformers = {'ft': ft, 'ct': ct}
lgb_model = ClassifierModel(model_wrapper=LGBWrapper())
lgb_model.fit(X=data, y=y, folds=folds, params=param, preprocesser=mt, transformers=transformers,
                    eval_metric='cappa', cols_to_drop=cols_to_drop)

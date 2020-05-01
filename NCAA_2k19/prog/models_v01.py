
# %% INITIALIZATION ------------------------------------------------------------

import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir('/home/marco/Documents/Lavori/Kaggle/NCAA_male_2019')

from ncaa19 import *

sns.set()

#-------------------------------------------------------------------------------

# %% LOAD DATA -----------------------------------------------------------------

PATH_DATASETS = '/home/marco/Documents/Lavori/Kaggle/NCAA_male_2019/input/'
PATH_OUTPUT = '/home/marco/Documents/Lavori/Kaggle/NCAA_male_2019/output/'
PATH_ELAB = '/home/marco/Documents/Lavori/Kaggle/NCAA_male_2019/elab/'

REGRESSORS  = ['delta_poss_m', 'delta_opp_poss_m',
              'delta_ass_ratio', 'delta_tov_ratio',
              'delta_reb_rate', 'delta_opp_true_fg_pct',
              'delta_off_rating_m', 'delta_def_rating_m',
              'delta_net_rating_m', 'delta_pace_m',
              'delta_off_rating_m_last30D', 'delta_def_rating_m_last30D',
              'delta_net_rating_m_last30D', 'delta_off_rating_m_vs_topseeds',
              'delta_def_rating_m_vs_topseeds', 'delta_net_rating_m_vs_topseeds',
              'delta_c_N_season', 'delta_w_pct',
              'delta_w_pct_last30D', 'delta_w_pct_vs_topseeds',
              'delta_c_W_PCT_allT', 'delta_c_W_PCT_vs_topseeds_allT',
              'delta_MOR', 'delta_POM',
              'delta_SAG']

df_rs_c_res = pd.read_csv(PATH_DATASETS + 'datafiles/RegularSeasonCompactResults.csv')
df_rs_d_res = pd.read_csv(PATH_DATASETS + 'datafiles/RegularSeasonDetailedResults.csv')
df_teams = pd.read_csv(PATH_DATASETS + 'datafiles/Teams.csv')
df_seeds = pd.read_csv(PATH_DATASETS + 'datafiles/NCAATourneySeeds.csv')
coaches = pd.read_csv(PATH_DATASETS + 'datafiles/TeamCoaches.csv')
df_tourn = pd.read_csv(PATH_DATASETS + 'datafiles/NCAATourneyCompactResults.csv')
mysub = pd.read_csv(PATH_DATASETS + 'SampleSubmissionStage1.csv')
massey = pd.read_csv(PATH_DATASETS + 'MasseyOrdinals/MasseyOrdinals.csv')

df_features_all = pd.read_csv(PATH_OUTPUT + 'NCAA_dataset_features.csv',
                              sep='|')

#-------------------------------------------------------------------------------

# %% ISOTONIC REGRESSION -------------------------------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

def ir_fit(var_series, target_series):
  ir = IsotonicRegression(increasing="auto")
  ir.fit(var_series, target_series)
  return(ir)

def ir_apply(ir, var_series):
  return(pd.Series(ir.predict(var_series), index = var_series.index))

def ir_plot(var_base, var_iso, var_target, n_classes = 10):
  cl_x = pd.qcut(var_base, n_classes, labels = False, duplicates = 'drop')
  stats_td = var_target.groupby(cl_x).mean()
  stats_iso = var_iso.groupby(cl_x).mean()
  # plot figure
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(stats_td, 'bs-')
  ax.plot(stats_iso, 'rs-')
  ax.yaxis.grid(True, linestyle='--', linewidth=1)
  ax.xaxis.set_label_text("Percentili variabile X")
  ax.yaxis.set_label_text("Td")
  plt.show()

filter_train = (df_features_all['Season']>=2003) & (df_features_all['Season']<=2014)
filter_test = (df_features_all['Season']==2015)
df_train = df_features_all.loc[filter_train, :].copy()
df_test = df_features_all.loc[filter_test, :].copy()
df_train.fillna(0, inplace=True) # is it right... @Stefano
df_test.fillna(0, inplace=True)

# apply isotonic transformation
for regressor in REGRESSORS:

    print("Transform feature {0}".format(regressor))

    ir_var = ir_fit(df_train[regressor], df_train['win_dummy'])
    df_train['piso_' + regressor] =\
      ir_apply(ir_var, df_train[regressor])
    df_test['piso_' + regressor] =\
      ir_apply(ir_var, df_test[regressor])

    ir_plot(df_test[regressor],
            df_test['piso_' + regressor],
            df_test['win_dummy'], 10)

# pd.crosstab(pd.isna(df_train['delta_off_rating_m_vs_topseeds']), 1)

#-------------------------------------------------------------------------------

# %% REGRESSION PT. 1 ----------------------------------------------------------

import numpy as np
from itertools import chain

def logloss(y_hat, p_1win):
    n = len(y_hat)
    logloss = (-1/n) * sum((y_hat * np.log(p_1win)) + (1 - y_hat) * np.log(1 - p_1win))
    return(logloss)

def logistic(df_tr, features, target, df_valid=None, penalty='l2', C=1.0):
    X = df_tr.loc[:, features]
    y = df_tr.loc[:, target]
    sk_model = LogisticRegression(fit_intercept=True, penalty=penalty, C=C)\
        .fit(X, y)
    df_results = pd.DataFrame({'feature': ["(Intercept)"] + features,
                               'beta': sk_model.intercept_.tolist() + sk_model.coef_[0].tolist()})
    print(df_results)
    if df_valid is not None:
        X_valid = df_valid.loc[:, features]
        y_valid = df_valid.loc[:, target]
        X_valid.fillna(0, inplace=True) # TODO: ask Stefano!
        p_valid = pd.Series(sk_model.predict(X_valid), index = y_valid.index)
        p_valid[p_valid >= 0.97] = 0.97
        p_valid[p_valid <= 0.03] = 0.03
        print(max(p_valid))
        print(min(p_valid))
        print("Log loss for test: {0}".format(logloss(y_valid, p_valid)))

    return None

logistic(df_train, REGRESSORS,
         'win_dummy', df_test, penalty='l1', C=0.50)

# -.10 log loss... yet still very high!
logistic(df_train, ['piso_' + r for r in REGRESSORS],
         'win_dummy', df_test, penalty='l1', C=0.50)



#-------------------------------------------------------------------------------


# %% PARAMETERS ----------------------------------------------------------------

PATH_DATASETS = '<PATH TO DATASETS>'
PATH_PROG = '<PATH TO PROG>'

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

#-------------------------------------------------------------------------------
              
# %% INITIALIZATION ------------------------------------------------------------

import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir(PATH_PROG)

from ncaa19 import *

sns.set()

#-------------------------------------------------------------------------------

# %% LOAD DATA -----------------------------------------------------------------

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

df_train = df_features_all.loc[df_features_all['Season']>=2003, :].copy()

id_regressor = 2
ir_var = ir_fit(df_train[REGRESSORS[id_regressor]], df_train['win_dummy'])
df_train['piso_' + REGRESSORS[id_regressor]] =\
  ir_apply(ir_var, df_train[REGRESSORS[id_regressor]])

ir_plot(df_train[REGRESSORS[id_regressor]],
        df_train['piso_' + REGRESSORS[id_regressor]],
        df_train['win_dummy'],
        20)
ir_plot(df_train[REGRESSORS[id_regressor]],
        df_train['piso_' + REGRESSORS[id_regressor]],
        df_train['win_dummy'],
        10)


#-------------------------------------------------------------------------------



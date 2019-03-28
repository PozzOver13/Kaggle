# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os as os
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

from matplotlib import pyplot as plt
%matplotlib inline
import matplotlib as mpl
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib import rcParams
#rcParams['figure.figsize'] = 20, 10
plt.style.use('ggplot')

os.chdir('/Users/stefanopozzati/Documents/GitHub/data_science/ncaa_2018/')

seeds = pd.read_csv(
    '/Users/stefanopozzati/Documents/GitHub/data_science/ncaa_2018/datainput/DataFiles/NCAATourneySeeds.csv')
tourney_results = pd.read_csv(
    '/Users/stefanopozzati/Documents/GitHub/data_science/ncaa_2018/datainput/DataFiles/NCAATourneyCompactResults.csv')
regular_results = pd.read_csv(
    '/Users/stefanopozzati/Documents/GitHub/data_science/ncaa_2018/datainput/DataFiles/RegularSeasonCompactResults.csv')

# get information
tourney_results.info()

# view first lines
tourney_results.head()

# create the variable which describe the difference betweeen scores
tourney_results["W-LScore"] = tourney_results["WScore"] - tourney_results["LScore"]

# count data
df = tourney_results

print(df['DayNum'].describe())
df.groupby('Season', 'DayNum').count()

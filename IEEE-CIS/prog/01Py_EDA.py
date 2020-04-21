#-----------------------------------------------------------------------------#
# PROGRAMMA: 01Py_IEEE-ICS_EDA.py
# DATA:      22-08-2019
# NOTE:      esploro le variabili
#-----------------------------------------------------------------------------#


#### INIZIALIZZAZIONE ####
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

folder_path = 'C:/Users/cg08900/Documents/GitHub/Kaggle/IEEE-CIS/'

#### CARICAMENTO DATI INPUT ####
train_identity = pd.read_csv(f'{folder_path}datainput/train_identity.csv')
train_transaction = pd.read_csv(f'{folder_path}datainput/train_transaction.csv')
test_identity = pd.read_csv(f'{folder_path}datainput/test_identity.csv')
test_transaction = pd.read_csv(f'{folder_path}datainput/test_transaction.csv')
# let's combine the data and work with the whole dataset
train = pd.merge(train_transaction, 
                 train_identity, 
                 on='TransactionID', 
                 how='left')
test = pd.merge(test_transaction, 
                test_identity, 
                on='TransactionID', 
                how='left')

del train_identity, train_transaction, test_identity, test_transaction

#### DATA QUALITY ####
stats = train.describe


#### MODELLO ####

#### ALGORITMO ####

#### SAVE ####
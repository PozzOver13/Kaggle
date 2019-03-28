#--------------------------------------------------------------------------------------------------------#
# PROGRAMMA: misuro performance
# DATA:      07/02/2019
# NOTE:      
#--------------------------------------------------------------------------------------------------------#

rm(list=ls())
for (i in 1:10) gc()

#### INIZIALIZZAZIONE ####
require(tidyverse)

#### CARICAMENTO DATI ####
df_in <- read.csv("C:/Users/cg08900/Documents/Pandora/Personale/kaggle/NCAA_2k19/elaborazioni/01Py_dataset_per_metriche.csv", sep = "|")

source(paste0("C:/Users/cg08900/Documents/Pandora/Lavoro/Locale/BPM/prog/functions/", "calcolo_accuracy.R"))

#### AR ####
df_in %>%
  select(-X) %>%
  rename(STATUS_01 =win_dummy) %>%
  gather("var", "valore", -STATUS_01) %>% 
  group_by(var) %>% 
  summarise(N = n(),
            Ndef = sum(STATUS_01, na.rm = T),
            AR = somers2(x = valore, y = STATUS_01)["Dxy"],
            C = somers2(x = valore, y = STATUS_01)["C"])

#--------------------------------------------------------------------------------------------------------#
# PROGRAMMA: 07_QUORA_analisi_distribuzioni.R
# DATA:      05-02-2019
# NOTE:      Kaggle - Quora - analisi distribuzioni
#--------------------------------------------------------------------------------------------------------#


#### INIZIALIZZAZIONE ####

require(dplyr)
require(tidyverse)

setwd("C:/users/cg08900/Documents/Pandora/Personale/kaggle/Quora/datasets/")


df_in <- read.csv("validazione_con_pr_xgboost.csv", sep = "|")

df_in %>% 
  filter(target == 1) %>%
  select(starts_with("pr_")) %>%
  gather("pr_stimata", "valore", -pr_is_1) %>%
  ggplot(aes(valore, fill = pr_stimata)) +
  geom_density() +
  facet_grid(pr_is_1~pr_stimata, scales = "free_y") +
  theme_bw() +
  labs(title = "Distribuzione pr stimate | target = 1")

df_in %>% 
  filter(target == 0) %>%
  select(starts_with("pr_")) %>%
  gather("pr_stimata", "valore", -pr_is_1) %>%
  ggplot(aes(valore, fill = pr_stimata)) +
  geom_density() +
  facet_grid(pr_is_1~pr_stimata, scales = "free_y") +
  theme_bw() +
  labs(title = "Distribuzione pr stimate | target = 0")

## Correlation
cor(df_in %>% 
      select(starts_with("pr_")))

cor(df_in %>% 
      filter(target == 1) %>%
      select(starts_with("pr_")))

cor(df_in %>% 
      filter(target == 0) %>%
      select(starts_with("pr_")))

cor(df_in %>% 
      filter(target == 1 & pr_is_1 == 0) %>%
      select(starts_with("pr_")) %>%
      select(-pr_is_1))

cor(df_in %>% 
      filter(target == 0 & pr_is_1 == 1) %>%
      select(starts_with("pr_")) %>%
      select(-pr_is_1))
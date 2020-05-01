########################################################
# PROGRAM: O2R NCAA
# DATE:    2018-03-01
# NOTE:    woe transformation
########################################################

rm(list = ls())
for (i in 1:10) gc()

#### SET UP #### 
# packages
require(tidyverse)
require(woe)

# personal ggplot theme
source("functions/pozzover_theme.R")
source("functions/compute_woe_1.R")

load("datasets/02R_dataset_training_2008_2013.rdata")

# selecting the potential predictors
df = training %>%
  select(starts_with("delta"), target) %>%
  as.data.frame()

names_var = names(df)[1:(dim(df)[2]-1)]

for (i in 1:length(names_var)) {
  
  df_woe = compute_woe(df_in = df, 
                       var = names_var[i], 
                       n_class = 10)
  
  df_woe$feature = names_var[i]
  
  df[[paste0("lgt_", names_var[i])]] =
    as.numeric(as.character(cut(df[[names_var[i]]], 
        breaks = c(df_woe$MIN, max(df_woe$MAX)),
        labels = df_woe$WOE,
        include.lowest = T)))
  
  if (i == 1) {
    df_woe_final = df_woe
  } else {
    df_woe_final = df_woe_final %>% bind_rows(df_woe)
  }
}

df %>% 
  summarise_all(funs(sum(is.na(.)))) %>%
  gather("var", "missing")

check = df %>%
  head()

df_expl_woe = df %>%
  select(starts_with("lgt"))

#### SAVE ####
save(df_expl_woe,
     file = "datasets/02R_explanatory_woe.rdata")
save(df_woe_final,
     file = "datasets/02R_woe_classes.rdata")

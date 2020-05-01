########################################################
# PROGRAM: 11R NCAA
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

load("datasets/10R_dataset_training_2008_2017.rdata")

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

df_expl_woe = df %>%
  select(starts_with("lgt"))

df_expl_woe %>%
  summarise_all(funs(max(.))) %>%
  gather("var", "max") %>%
  arrange(max)

df_expl_woe %>%
  summarise_all(funs(min(.))) %>%
  gather("var", "min") %>%
  arrange(min)

df_expl_woe = df_expl_woe %>% 
  mutate_all(funs(ifelse(. > 200, 
                         210,
                         .)))

df_expl_woe = df_expl_woe %>% 
  mutate_all(funs(ifelse(. < -300, 
                         -310,
                         .)))

#### SAVE ####
save(df_expl_woe,
     file = "datasets/11R_explanatory_woe.rdata")
save(df_woe_final,
     file = "datasets/11R_woe_classes.rdata")

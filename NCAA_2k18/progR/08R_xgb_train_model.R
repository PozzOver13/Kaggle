########################################################
# PROGRAM: O2R NCAA
# DATE:    2018-03-01
# NOTE:    creating score
#          remark => target = W = 1
#          negative logit == higher probability to lose
#          positive logit == higher probability to win
########################################################

rm(list = ls())
for (i in 1:10) gc()

#### SET UP #### 
# packages
require(caret)
require(xgboost)
require(tidyverse)
require(stringi)

# personal ggplot theme
source("functions/pozzover_theme.R")

# logloss metric function
log_loss = function(pred, actual) {
  -1*mean(log(pred[model.matrix(~ actual + 0) - pred > 0]))
}

#### DATA IMPORT ####
load("datasets/02R_dataset_training_2008_2013.rdata")
load("datasets/02R_explanatory_woe.rdata")

# adjusting the training dataset with logit Xs
training = training %>%
  select(-starts_with("delta")) %>%
  bind_cols(df_expl_woe)

#### XGBOOST ####
# List of  variables with continuous values 

# Train and Test split based on 70% and 30%
set.seed(123)
train_data = training %>%
  select(contains("_topseeds"), contains("nrm_"), target) %>%
  select(contains("delta"), contains("lgt_delta_nrm_"), target)

# Custom functions for calculation of Precision and Recall
frac_trzero = (table(train_data$target)[[1]])/nrow(train_data)
frac_trone = (table(train_data$target)[[2]])/nrow(train_data)


prec_zero <- function(act,pred){  tble = table(act,pred)
return( round( tble[1,1]/(tble[1,1]+tble[2,1]),4)  ) }

prec_one <- function(act,pred){ tble = table(act,pred)
return( round( tble[2,2]/(tble[2,2]+tble[1,2]),4)   ) }

recl_zero <- function(act,pred){tble = table(act,pred)
return( round( tble[1,1]/(tble[1,1]+tble[1,2]),4)   ) }

recl_one <- function(act,pred){ tble = table(act,pred)
return( round( tble[2,2]/(tble[2,2]+tble[2,1]),4)  ) }

accrcy <- function(act,pred){ tble = table(act,pred)
return( round((tble[1,1]+tble[2,2])/sum(tble),4)) }

y = train_data$target

# XGBoost Classifier Training
xgb <- xgboost(data = data.matrix(train_data[,-15]),
               label = y,
               eta = 0.04,
               max_depth = 2, 
               nround=2000, 
               subsample = 0.5, 
               colsample_bytree = 0.5, 
               seed = 1, 
               eval_metric = "logloss", 
               objective = "binary:logistic",
               nthread = 3)

xgb.importance(train_data[-15] %>% names(),
               xgb) %>%
  ggplot(aes(reorder(Feature, -Gain), Gain)) +
  geom_bar(stat = "identity", 
           alpha = 0.5, 
           fill = "lightseagreen") +
  coord_flip() +
  labs(title = "XGB - importance",
       x = "") +
  theme_pozzover

#### SAVE ####
save(xgb, file = "progR/08R_xgb_model.rdata")

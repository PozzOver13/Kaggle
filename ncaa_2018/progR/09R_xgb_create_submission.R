########################################################
# PROGRAM: O2R NCAA
# DATE:    2018-03-01
# NOTE:    test model application woe
########################################################

rm(list = ls())
for (i in 1:10) gc()

#### SET UP #### 
require(tidyverse)

source("functions/pozzover_theme.R")
source("functions/compute_woe_1.R")

# logloss metric function
log_loss = function(pred, actual){
  -1*mean(log(pred[model.matrix(~ actual + 0) - pred > 0]))
}

#### DATA IMPORT ####
rs_results = 
  read_csv("datainput/DataFiles/RegularSeasonCompactResults.csv")
rs_results_plus = 
  read_csv("datainput/DataFiles/RegularSeasonDetailedResults.csv")
seeds = 
  read_csv("datainput/DataFiles/NCAATourneySeeds.csv")
teams = 
  read_csv("datainput/DataFiles/Teams.csv")
coaches = 
  read_csv("datainput/DataFiles/TeamCoaches.csv")
mm_results = 
  read_csv("datainput/DataFiles/NCAATourneyCompactResults.csv")
submission = 
  read_csv("datainput/DataFiles/SampleSubmissionStage1.csv")

load("datasets/01R_teams_with_advanced_metrics.rdata")

load(file = "progR/08R_xgb_model.rdata")

load(file = "datasets/02R_woe_classes.rdata")

my_sub = submission %>%
  mutate(Season = as.numeric(str_split_fixed(ID, "_", 3)[,1]),
         LF_TeamID = as.numeric(str_split_fixed(ID, "_", 3)[,2]),
         RT_TeamID = as.numeric(str_split_fixed(ID, "_", 3)[,3])) 

# dataset explanatory variables
df_Xs = df_teams_plus_final %>% 
  select(TeamID, Season,
         poss_m, 
         opp_poss_m,
         ass_ratio_m,
         tov_ratio_m,
         reb_rate_m,
         opp_true_fg_pct_m,
         off_rating_m, def_rating_m, net_rating_m,
         pace_m,
         off_rating_m_last30D,
         def_rating_m_last30D,
         net_rating_m_last30D,
         poss_m_vs_topseeds, 
         opp_poss_m_vs_topseeds,
         ass_ratio_m_vs_topseeds,
         tov_ratio_m_vs_topseeds,
         reb_rate_m_vs_topseeds,
         opp_true_fg_pct_m_vs_topseeds,
         off_rating_m_vs_topseeds,
         def_rating_m_vs_topseeds,
         net_rating_m_vs_topseeds,
         W_PCT, 
         W_PCT_last30D,
         W_PCT_vs_topseeds,
         seed_n,
         nrm_c_wins, 
         nrm_c_top_wins, 
         nrm_c_season, 
         nrm_c_champ_wins) 

# class creation  
#mutate(seed_class = ifelse(is.na(seed_n), "no_seed_class",
#                           ifelse(seed_n %in% 1:5,
#                                  "first_seed_class",
#                                  "second_seed_class")),
#       seed_region_class = ifelse(is.na(seed_region), "ND", seed_region))

# function to create the training dataset
# standardized in order to suitable to next years
create_df_to_subs = function(df_sub, df_Xs) {
  
  # construct training dataset
  mm_resultsdet_out = df_sub %>%
    left_join(df_Xs %>%
                rename(LF_TeamID = TeamID), 
              by = c("Season", "LF_TeamID")) %>%
    left_join(df_Xs %>%
                rename(RT_TeamID = TeamID), 
              by = c("Season", "RT_TeamID"))
  
  names(mm_resultsdet_out) = 
    str_replace_all(names(mm_resultsdet_out), "\\.", "_")
  
  create_delta = function(df, xs, ys) {
    df[[xs]] - df[[ys]]
  }
  
  var_list_to_delta = names(df_Xs)[3:(dim(df_Xs)[2])]
  
  # calculate every dataset
  for (i in 1:length(var_list_to_delta)) {
    
    delta = paste0("delta_", var_list_to_delta[i])
    var_x = paste0(var_list_to_delta[i], "_x")
    var_y = paste0(var_list_to_delta[i], "_y")
    
    mm_resultsdet_out[[delta]] <- create_delta(mm_resultsdet_out,
                                               var_x, var_y)
  }
  
  # mm_resultsdet_out$target = ifelse(mm_resultsdet_out$WTeamID > 
  #                                     mm_resultsdet_out$LTeamID, 0, 1)
  
  mm_resultsdet_out
}

# apply function
my_sub_top = create_df_to_subs(my_sub, df_Xs)

# selecting the potential predictors
df = my_sub_top %>%
  select(starts_with("delta")) %>%
  as.data.frame() %>%
  mutate_all(funs(ifelse(is.na(.),
                         median(., na.rm = T),
                         .)))

names_var = names(df)[1:(dim(df)[2])]

# create woe
for (i in 1:length(names_var)) {
  
  df_woe = df_woe_final %>%
    filter(feature == names_var[i])
  
  df_woe$MIN[1] <- -Inf
  df_woe$MAX[dim(df_woe)[1]] <- Inf
  
  df[[paste0("lgt_", names_var[i])]] =
    as.numeric(as.character(cut(df[[names_var[i]]], 
                                breaks = c(df_woe$MIN, max(df_woe$MAX)),
                                labels = df_woe$WOE,
                                include.lowest = T)))
}

df %>% 
  summarise_all(funs(sum(is.na(.)))) %>%
  gather("var", "missing")

df_expl_woe = df %>%
  select(starts_with("lgt"))

# adjusting the training dataset with logit Xs
my_sub_top = my_sub_top %>%
  select(-starts_with("delta")) %>%
  bind_cols(df_expl_woe)

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

test_data = my_sub_top %>%
  select(starts_with("lgt"))

ts_y_pred_prob <- predict(xgb, data.matrix(test_data))


#### SAVE #### 
my_sub_stage1 = my_sub_top %>%
  select(1) %>%
  mutate(Pred = ts_y_pred_prob)
names(my_sub_stage1) <- c("ID", "Pred")

write.table(my_sub_stage1, 
            file = "datasets/09R_submission_stage1_xgb.csv", 
            sep = ",",
            row.names = F)

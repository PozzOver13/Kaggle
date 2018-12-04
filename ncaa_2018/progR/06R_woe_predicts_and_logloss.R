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
  read_csv("datainput/sampe.csv")

load("datasets/01R_teams_with_advanced_metrics.rdata")

load(file = "progR/02R_woe_logistic_model_coach_score.rdata")
load(file = "progR/02R_woe_logistic_model_team_score.rdata")
load(file = "progR/02R_woe_logistic_model_marchmadness_score.rdata")

# selecting from 2008 to 2013 in order to have historical data 
# without considering too much old results
mm_201417 = mm_results %>% 
  filter(Season %in% 2014:2017)

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
create_df_training = function(mm_resultsdet, df_Xs) {
  
  # construct training dataset
  mm_resultsdet_out = mm_resultsdet %>%
    mutate(LF_TeamID = pmin(WTeamID, LTeamID),
           RT_TeamID = pmax(WTeamID, LTeamID)) %>%
    left_join(df_Xs %>%
                rename(LF_TeamID = TeamID), 
              by = c("Season", "LF_TeamID")) %>%
    left_join(df_Xs %>%
                rename(RT_TeamID = TeamID), 
              by = c("Season", "RT_TeamID"))
  names(mm_resultsdet_out) = str_replace_all(names(mm_resultsdet_out), "\\.", "_")
  
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
  
  mm_resultsdet_out$target = ifelse(mm_resultsdet_out$WTeamID > 
                                      mm_resultsdet_out$LTeamID, 0, 1)
  
  mm_resultsdet_out
}

# apply function
testing = create_df_training(mm_201417, df_Xs)

testing = testing %>%
  mutate_at(vars(starts_with("delta")),
            funs(ifelse(is.na(.), 
                        mean(., na.rm = T),
                        .)))

# selecting the potential predictors
df = testing %>%
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

# adjusting the training dataset with logit Xs
testing = testing %>%
  select(-starts_with("delta")) %>%
  bind_cols(df_expl_woe)

model_lists = list(coach_score_model_final, 
                   team_score_model_final, 
                   mad_score_model_final)

#### TESTING ####
apply_logistic_model <- function(df_in, model_list) {
  
  df_in %>% 
    mutate(coach_score = predict(model_list[[1]], newdata = df_in),
           team_score = predict(model_list[[2]], newdata = df_in)) %>%
    mutate(mad_score = predict(model_list[[3]], newdata = .),
           wins_pred = predict(model_list[[3]], 
                               newdata = ., type = "response"))
  
}

testing_score = apply_logistic_model(testing, model_lists)

# check log loss metric
log_loss(testing_score$wins_pred,
         testing_score$target)

# check log loss metric by year
season_test = 2014:2017

for (i in 1:length(season_test)) {
  
  ll = log_loss(
    testing_score$wins_pred[testing_score$Season == season_test[i]],
    testing_score$target[testing_score$Season == season_test[i]])
  
  print(paste0("season ", season_test[i], 
               "logloss: ", round(ll, 5)))
}






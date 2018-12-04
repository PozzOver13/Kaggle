########################################################
# PROGRAM: O2R NCAA
# DATE:    2018-03-01
# NOTE:    multivariate analysis
########################################################

rm(list = ls())
for (i in 1:10) gc()

#### SET UP #### 
require(tidyverse)
require(rpart)

source("functions/pozzover_theme.R")

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

load("datasets/01R_teams_with_advanced_metrics.rdata")
load("datasets/02R_univ_variables.rdata")

# selecting from 2008 to 2013 in order to have historical data 
# without considering too much old results
mm_200813 = mm_results %>% 
  filter(Season %in% 2008:2013)

# dataset explanatory variables
df_Xs = df_teams_plus_final %>% 
  select(TeamID, Season,
         poss_m, opp_poss_m,
         off_rating_m, def_rating_m, net_rating_m,
         pace_m,
         off_rating_m_last30D,
         def_rating_m_last30D,
         net_rating_m_last30D,
         off_rating_m_vs_topseeds,
         def_rating_m_vs_topseeds,
         net_rating_m_vs_topseeds,
         W_PCT, 
         W_PCT_last30D,
         W_PCT_vs_topseeds,
         seed_n,
         coach_score,
         team_score) 

# check duplicates
dups  = df_teams_plus_final %>%
  group_by(TeamID, Season) %>%
  mutate(N = n()) %>%
  filter(N > 1)

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
training = create_df_training(mm_200813, df_Xs)

#### MULTIVARIATE ANALYSIS ####
# selecting only strongest explanatory variables
var_train = c("target", df_univ_reg_sign$term)
train_set = training[, var_train]

# original multivariate model
summary(glm(target~., 
    data = train_set, 
    family = "binomial"))

# objective multivariate model
summary(glm(target~delta_coach_score + delta_team_score, 
            data = train_set, 
            family = "binomial"))

# original multivariate model
predictions_lgt = glm(target~delta_coach_score + delta_team_score, 
                             data = train_set, 
                             family = "binomial") %>%
  broom::augment(type.predict = "link") %>% 
  mutate(y_hat = .fitted)

predictions_pr = glm(target~delta_coach_score + delta_team_score, 
                  data = train_set, 
                  family = "binomial") %>%
  broom::augment(type.predict = "response") %>% 
  mutate(y_hat = .fitted)

head(predictions_lgt[, 1:5])
head(predictions_pr[, 1:5])





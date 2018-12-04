########################################################
# PROGRAM: O2R NCAA
# DATE:    2018-03-01
# NOTE:    univariate analysis
########################################################

rm(list = ls())
for (i in 1:10) gc()

#### SET UP #### 
require(tidyverse)

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

# selecting from 2008 to 2013 in order to have historical data 
# without considering too much old results
mm_200813 = mm_results %>% 
  filter(Season %in% 2008:2013)

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
         nrm_c_champ_wins,
         nrm_MOR,
         nrm_POM,
         nrm_SAG) 

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
training = create_df_training(mm_200813, df_Xs)

# imputing missing values with the mean
training = training %>%
  mutate_all(funs(ifelse(is.na(.), 
                         mean(., na.rm = T),
                         .)))

#### UNIVARIATE ANALYSIS ####
## Graphical analysis in order to select the best predictors
training %>%
  filter(Season == 2011) %>%
  ggplot(aes(delta_nrm_MOR, target)) +
  geom_point() +
  geom_smooth(method = "glm", 
              method.args = list(family = "binomial"))

# univariate variable list
vars_univ = names(training)[str_detect(names(training), "^delta")]

range_deltas = training[, vars_univ] %>%
  summarise_all(funs(max(., na.rm = T) - min(., na.rm = T))) %>%
  gather("vars", "range") %>%
  arrange(range)

# divide in order to face the different scale
vars_univ_1 = range_deltas$vars[1:9]
vars_univ_2 = range_deltas$vars[10:30]

# create 2 datset in order to have more comunicative graphs
create_dataset_for_univariate_plot = function(df_in, var_list) {
  for (i in 1:length(var_list)) {
    
    df_univ_plot = df_in[, c("target", var_list[i])]
    df_univ_plot$var <- var_list[i]
    names(df_univ_plot) <- c("target", "x", "vars")
    
    if (i == 1) {
      df_out = df_univ_plot
    } else {
      df_out = bind_rows(df_out, df_univ_plot)
    }
  }
  df_out
}

df_univ_plot_final_1 = create_dataset_for_univariate_plot(training, vars_univ_1)
df_univ_plot_final_2 = create_dataset_for_univariate_plot(training, vars_univ_2)

# plot delta vs target (ratings)
df_univ_plot_final_1 %>%
  ggplot(aes(x, target)) +
  geom_point() +
  geom_smooth(method = "glm", 
              method.args = list(family = "binomial")) +
  facet_wrap(~vars) +
  labs(title = "Univariate relationship score and %") +
  theme_pozzover

# plot delta vs target (score & w percentage)
df_univ_plot_final_2 %>%
  ggplot(aes(x, target)) +
  geom_point() +
  geom_smooth(method = "glm", 
              method.args = list(family = "binomial")) +
  facet_wrap(~vars) +
  labs(title = "Univariate relationship ratings") +
  theme_pozzover

## Regression
for (i in 1:length(vars_univ)) {
  
  form = paste0("target~", vars_univ[i])
  df_univ_reg = broom::tidy(glm(form, 
                                   data = training, 
                                   family = "binomial"))
  
  if (i == 1) {
    df_univ_reg_final = df_univ_reg
  } else {
    df_univ_reg_final = bind_rows(df_univ_reg_final, 
                                  df_univ_reg)
  }

}

df_univ_reg_final = df_univ_reg_final %>%
  rename(beta=estimate,
         p_value=p.value) %>%
  select(term, beta, p_value) %>%
  mutate(significant = p_value < 0.005,
         p_value = round(p_value, 4))

df_univ_reg_sign = df_univ_reg_final %>%
  filter(significant == T) 

pot_model_vars = df_univ_reg_sign$term

#### SAVE ####
save(df_univ_reg_sign, 
     file = "datasets/02R_univ_variables.rdata")
save(training, 
     file = "datasets/02R_dataset_training_2008_2013.rdata")

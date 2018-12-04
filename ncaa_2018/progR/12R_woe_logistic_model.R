########################################################
# PROGRAM: 12R NCAA
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
require(tidyverse)
require(stringi)

# personal ggplot theme
source("functions/pozzover_theme.R")

# logloss metric function
log_loss = function(pred, actual){
  -1*mean(log(pred[model.matrix(~ actual + 0) - pred > 0]))
}

#### DATA IMPORT ####
load("datasets/10R_dataset_training_2008_2017.rdata")
load("datasets/11R_explanatory_woe.rdata")

# adjusting the training dataset with logit Xs
training = training %>%
  select(-starts_with("delta")) %>%
  bind_cols(df_expl_woe)

#### COACH SCORE ####
# coach score predictors
var_coach_score = c("lgt_delta_nrm_c_wins", "lgt_delta_nrm_c_top_wins",
                    "lgt_delta_nrm_c_season", "lgt_delta_nrm_c_champ_wins")
# creating all combination of predictors
regMat <- expand.grid(c(TRUE,FALSE), c(TRUE,FALSE),
                      c(TRUE,FALSE), c(TRUE,FALSE))
names(regMat) <- c("lgt_delta_nrm_c_wins", "lgt_delta_nrm_c_top_wins",
                   "lgt_delta_nrm_c_season", "lgt_delta_nrm_c_champ_wins")

model_list = vector("list", dim(regMat)[1]-1) 
loglossmodels = vector("list", dim(regMat)[1]-1)
n_vars_list = vector("list", dim(regMat)[1]-1)
n_vars_sign_list = vector("list", dim(regMat)[1]-1)
valid_model = vector("list", dim(regMat)[1]-1)

# loop in order to understand which predictors are more valuable 
# in order to reduce the logloss metric
for (i in 1:length(loglossmodels)) {
  
  slct_vars = regMat[i,] %>% as.vector() %>% as.logical()
  
  form_coach_score = paste0("target~",
                            stri_flatten(var_coach_score[slct_vars], "+"), "-1")
  
  predict_games = glm(as.formula(form_coach_score), 
                      data = training, 
                      family = "binomial") %>% 
    broom::augment(type.predict = "response") %>% 
    mutate(y_hat = .fitted) 
  
  n_vars = glm(as.formula(form_coach_score), 
               data = training, 
               family = "binomial") %>% 
    broom::tidy() %>% 
    pull(p.value) 
  
  betas = glm(as.formula(form_coach_score), 
              data = training, 
              family = "binomial") %>% 
    coefficients()
  
  
  loglossmodels[[i]] <- log_loss(predict_games$y_hat,
                                 predict_games$target)
  
  model_list[[i]] <- stri_flatten(var_coach_score[slct_vars], " ")
  n_vars_list[[i]] <- length(var_coach_score[slct_vars])
  n_vars_sign_list[[i]] <- sum(n_vars < 0.05) / length(n_vars)
  valid_model[[i]] <- all(!is.na(betas))
}
# summarizing results into a single dataframe
df_coach_model_results = data.frame(model_id = seq(1, 15, 1),
                                    model_descr = model_list %>% unlist(),
                                    model_n_var = n_vars_list %>% unlist(),
                                    model_n_var_sign = n_vars_sign_list %>% unlist(),
                                    model_logloss = loglossmodels %>% unlist(),
                                    model_valid = valid_model %>% unlist()) %>%
  filter(model_valid) %>%
  arrange(model_logloss)


# visualizing the final results
df_coach_model_results %>%
  ggplot(aes(model_id, model_logloss)) +
  geom_point(col = "navajowhite3") +
  labs(title = "Logloss evolution through different models") +
  theme_pozzover

# idenitify the best model (manually)
# id_model = 
#   df_coach_model_results$model_id[df_coach_model_results$model_logloss ==
#                                     min(df_coach_model_results$model_logloss)]
id_model = 10

# calculating the ultimate score
slct_vars = regMat[id_model,] %>% as.vector() %>% as.logical()

form_coach_score = paste0("target~",
                          stri_flatten(var_coach_score[slct_vars], "+"), "-1")

glm(as.formula(form_coach_score), 
    data = training, 
    family = "binomial") %>% 
  summary()

coach_score_model_final = glm(as.formula(form_coach_score), 
                              data = training, 
                              family = "binomial")
df_coach_score_model_final = coach_score_model_final %>% 
  broom::augment() %>% 
  mutate(coach_score = .fitted) 

training_score = training %>%
  bind_cols(df_coach_score_model_final %>%
              select(coach_score))

#### TEAM SCORE ####
# team score predictors
deltas = names(training_score)[names(training_score) %>% 
                                 str_detect("^lgt_")]
var_team_score = deltas[deltas %>% 
                          str_detect("_vs_topseeds")]

# creating all combination of predictors
regMat <- expand.grid(c(TRUE,FALSE), c(TRUE,FALSE),
                      c(TRUE,FALSE), c(TRUE,FALSE),
                      c(TRUE,FALSE), c(TRUE,FALSE),
                      c(TRUE,FALSE), c(TRUE,FALSE),
                      c(TRUE,FALSE), c(TRUE,FALSE))
names(regMat) <- var_team_score

model_list = vector("list", dim(regMat)[1]-1) 
loglossmodels = vector("list", dim(regMat)[1]-1)
n_vars_list = vector("list", dim(regMat)[1]-1)
n_vars_sign_list = vector("list", dim(regMat)[1]-1)
valid_model = vector("list", dim(regMat)[1]-1)

# loop in order to understand which predictors are more valuable 
# in order to reduce the logloss metric
for (i in 1:length(loglossmodels)) {
  
  slct_vars = regMat[i,] %>% as.vector() %>% as.logical()
  
  form_team_score = paste0("target~",
                           stri_flatten(var_team_score[slct_vars], "+"), "-1")
  
  predict_games = glm(as.formula(form_team_score), 
                      data = training, 
                      family = "binomial") %>% 
    broom::augment(type.predict = "response") %>% 
    mutate(y_hat = .fitted) 
  
  n_vars = glm(as.formula(form_team_score), 
               data = training, 
               family = "binomial") %>% 
    broom::tidy() %>% 
    pull(p.value)
  
  betas = glm(as.formula(form_team_score), 
              data = training, 
              family = "binomial") %>% 
    coefficients()
  
  loglossmodels[[i]] <- log_loss(predict_games$y_hat,
                                 predict_games$target)
  
  model_list[[i]] <- stri_flatten(var_team_score[slct_vars], " ")
  n_vars_list[[i]] <- length(var_team_score[slct_vars])
  n_vars_sign_list[[i]] <- sum(n_vars < 0.05) / length(n_vars)
  valid_model[[i]] <- all(!is.na(betas))
  
}
# summarizing results into a single dataframe
df_team_model_results = data.frame(model_id = seq(1, dim(regMat)[1]-1, 1),
                                   model_descr = model_list %>% unlist(),
                                   model_n_var = n_vars_list %>% unlist(),
                                   model_n_var_sign = n_vars_sign_list %>% unlist(),
                                   model_logloss = loglossmodels %>% unlist(),
                                   model_valid = valid_model %>% unlist()) %>%
  filter(model_valid) %>%
  arrange(model_logloss)

# visualizing the final results
df_team_model_results %>%
  ggplot(aes(model_id, model_logloss)) +
  geom_point(col = "navajowhite3") +
  labs(title = "Team Score - Logloss evolution through different models") +
  theme_pozzover

# idenitify the best model
# id_model = 
#   df_team_model_results$model_id[df_team_model_results$model_logloss ==
#                                        min(df_team_model_results$model_logloss)]
id_model = 773

# calculating the ultimate score
slct_vars = regMat[id_model,] %>% as.vector() %>% as.logical()

form_team_score = paste0("target~",
                         stri_flatten(var_team_score[slct_vars], "+"), "-1")

glm(as.formula(form_team_score), 
    data = training, 
    family = "binomial") %>% 
  summary()

team_score_model_final = glm(as.formula(form_team_score), 
                             data = training, 
                             family = "binomial")
df_team_score_model_final = team_score_model_final %>% 
  broom::augment() %>% 
  mutate(team_score = .fitted) 

training_score = training_score %>%
  bind_cols(df_team_score_model_final %>%
              select(team_score))

#### RANKING SCORE ####
# rank score predictors
deltas = names(training_score)[names(training_score) %>% 
                                 str_detect("^lgt_")]
var_rank_score = c("lgt_delta_nrm_MOR",
                   "lgt_delta_nrm_POM",
                   "lgt_delta_nrm_SAG")

# creating all combination of predictors
regMat <- expand.grid(c(TRUE,FALSE), c(TRUE,FALSE),
                      c(TRUE,FALSE))
names(regMat) <- var_rank_score

model_list = vector("list", dim(regMat)[1]-1) 
loglossmodels = vector("list", dim(regMat)[1]-1)
n_vars_list = vector("list", dim(regMat)[1]-1)
n_vars_sign_list = vector("list", dim(regMat)[1]-1)
valid_model = vector("list", dim(regMat)[1]-1)

# loop in order to understand which predictors are more valuable 
# in order to reduce the logloss metric
for (i in 1:length(loglossmodels)) {
  
  slct_vars = regMat[i,] %>% as.vector() %>% as.logical()
  
  form_rank_score = paste0("target~",
                           stri_flatten(var_rank_score[slct_vars], "+"), "-1")
  
  predict_games = glm(as.formula(form_rank_score), 
                      data = training, 
                      family = "binomial") %>% 
    broom::augment(type.predict = "response") %>% 
    mutate(y_hat = .fitted) 
  
  n_vars = glm(as.formula(form_rank_score), 
               data = training, 
               family = "binomial") %>% 
    broom::tidy() %>% 
    pull(p.value)
  
  betas = glm(as.formula(form_rank_score), 
              data = training, 
              family = "binomial") %>% 
    coefficients()
  
  loglossmodels[[i]] <- log_loss(predict_games$y_hat,
                                 predict_games$target)
  
  model_list[[i]] <- stri_flatten(var_rank_score[slct_vars], " ")
  n_vars_list[[i]] <- length(var_rank_score[slct_vars])
  n_vars_sign_list[[i]] <- sum(n_vars < 0.05) / length(n_vars)
  valid_model[[i]] <- all(!is.na(betas))
  
}
# summarizing results into a single dataframe
df_rank_model_results = data.frame(model_id = seq(1, dim(regMat)[1]-1, 1),
                                   model_descr = model_list %>% unlist(),
                                   model_n_var = n_vars_list %>% unlist(),
                                   model_n_var_sign = n_vars_sign_list %>% unlist(),
                                   model_logloss = loglossmodels %>% unlist(),
                                   model_valid = valid_model %>% unlist()) %>%
  filter(model_valid) %>%
  arrange(model_logloss)

# visualizing the final results
df_rank_model_results %>%
  ggplot(aes(model_id, model_logloss)) +
  geom_point(col = "navajowhite3") +
  labs(title = "Rank Score - Logloss evolution through different models") +
  theme_pozzover

# idenitify the best model
# id_model = 
#   df_team_model_results$model_id[df_team_model_results$model_logloss ==
#                                        min(df_team_model_results$model_logloss)]
id_model = 3

# calculating the ultimate score
slct_vars = regMat[id_model,] %>% as.vector() %>% as.logical()

form_rank_score = paste0("target~",
                         stri_flatten(var_rank_score[slct_vars], "+"), "-1")

glm(as.formula(form_rank_score), 
    data = training, 
    family = "binomial") %>% 
  summary()

rank_score_model_final = glm(as.formula(form_rank_score), 
                             data = training, 
                             family = "binomial")
df_rank_score_model_final = rank_score_model_final %>% 
  broom::augment() %>% 
  mutate(rank_score = .fitted) 

training_score = training_score %>%
  bind_cols(df_rank_score_model_final %>%
              select(rank_score))

#### MAD SCORE ####
# catchy name for the synthetic score
# MAD_SCORE = March Madness Score
var_mad_score = c("coach_score", "team_score", "rank_score")
form_mad_score = paste0("target~",
                        stri_flatten(var_mad_score, "+"), "-1")

# logistic model
glm(as.formula(form_mad_score), 
    data = training_score, 
    family = "binomial") %>% 
  summary()

# calculate the score
mad_score_model_final = glm(as.formula(form_mad_score), 
                            data = training_score, 
                            family = "binomial")
df_mad_score_model_final = mad_score_model_final %>% 
  broom::augment() %>% 
  mutate(mad_score = .fitted) 

# bind the new column
training_score = training_score %>%
  bind_cols(df_mad_score_model_final %>%
              select(mad_score))

# use the synthetic score as the final predictor 
var_mad_score = c("mad_score")
form_mad_score = paste0("target~",
                        stri_flatten(var_mad_score, "+"), "-1")

# logistic model
glm(as.formula(form_mad_score), 
    data = training_score, 
    family = "binomial") %>% 
  summary()

mad_score_model_final_sint = glm(as.formula(form_mad_score), 
                                 data = training_score, 
                                 family = "binomial")

df_train_mad_score_model_final = mad_score_model_final_sint %>% 
  broom::augment(type.predict = "response") %>% 
  mutate(w_probs_mad_score = .fitted)

# compute the logloss
loglossmodels_final_score <- 
  log_loss(df_train_mad_score_model_final$w_probs_mad_score,
           df_train_mad_score_model_final$target)

loglossmodels_final_score

## Graphical Analysis
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

df_univ_plot_final_1 = create_dataset_for_univariate_plot(training_score, 
                                                          c("coach_score", 
                                                            "team_score",
                                                            "mad_score"))

# plot delta vs target (ratings)
df_univ_plot_final_1 %>%
  ggplot(aes(x, target)) +
  geom_point() +
  geom_smooth(method = "glm", 
              method.args = list(family = "binomial")) +
  facet_wrap(~vars) +
  labs(title = "Univariate relationship score and %") +
  theme_pozzover

#### SAVE ####
coach_score_model_final %>% summary()
team_score_model_final %>% summary()
rank_score_model_final %>% summary()
mad_score_model_final %>% summary()

save(coach_score_model_final, 
     file = "progR/12R_woe_logistic_model_coach_score.rdata")
save(team_score_model_final, 
     file = "progR/12R_woe_logistic_model_team_score.rdata")
save(rank_score_model_final, 
     file = "progR/12R_woe_logistic_model_rank_score.rdata")
save(mad_score_model_final, 
     file = "progR/12R_woe_logistic_model_marchmadness_score.rdata")

save(training_score, 
     file = "datasets/12R_woe_training_dataset_2008_2017.rdata")
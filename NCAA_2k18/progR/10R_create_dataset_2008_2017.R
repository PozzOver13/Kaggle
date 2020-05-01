########################################################
# PROGRAM: 10R NCAA
# DATE:    2018-02-25
# NOTE:    creating training dataset
########################################################

rm(list = ls())
for (i in 1:10) gc()

#### SET UP #### 
require(tidyverse)

source("functions/pozzover_theme.R")

#### DATA IMPORT ####
rs_results = 
  read_csv("datainput/Stage2UpdatedDataFiles/RegularSeasonCompactResults.csv")
rs_results_plus = 
  read_csv("datainput/Stage2UpdatedDataFiles/RegularSeasonDetailedResults.csv")
seeds = 
  read_csv("datainput/Stage2UpdatedDataFiles/NCAATourneySeeds.csv")
teams = 
  read_csv("datainput/Stage2UpdatedDataFiles/Teams.csv")
coaches = 
  read_csv("datainput/DataFiles/TeamCoaches.csv")
mm_results = 
  read_csv("datainput/DataFiles/NCAATourneyCompactResults.csv")
mm_conf = 
  read_csv("datainput/DataFiles/TeamConferences.csv")
massey = 
  read_csv("datainput/MasseyOrdinals_thruSeason2018_Day128.csv")

#### FEATURE TRANSFORMATION ####
# clean seed information
seeds_cleaned = seeds %>% 
  select(TeamID, Season, Seed) %>%
  mutate(seed_n = str_sub(Seed, 2, -1),
         seed_n = as.numeric(str_replace_all(seed_n, "[a-z]", "")),
         seed_region = str_sub(Seed, 1, 1),
         top_seeded_teams = ifelse(!is.na(seed_n), 1, 0))

# create a dataset by team 
df_games = rs_results_plus %>% 
  mutate(poss = WFGA + 0.475*WFTA - WOR + WTO,
         opp_poss = LFGA + 0.475*LFTA - LOR + LTO,
         ass_ratio = (WAst*100)/(WFGA + (WFTA*0.44) + WAst + WTO),
         tov_ratio = (WTO*100)/(WFGA + (WFTA*0.44) + WAst + WTO),
         reb_rate = ((WOR+WDR)*200)/(40*(WOR+WDR+LOR+LDR)),
         opp_true_fg_pct = (LScore*50)/(LFGA + (LFTA*0.44)),
         off_rating = round((WScore / poss)*100, 2),
         def_rating = round((LScore / opp_poss)*100, 2), 
         net_rating = off_rating - def_rating,
         pace = 40*((poss+opp_poss)/(2*(200/5)))) %>%
  rename(TeamID = WTeamID,
         Score_left = WScore,
         Score_right = LScore,
         Loc = WLoc) %>%
  mutate(W_bin = 1) %>%
  select(TeamID,
         LTeamID,
         Season, 
         DayNum,
         Score_left,
         Score_right,
         Loc,
         W_bin,
         poss,
         opp_poss,
         ass_ratio,
         tov_ratio,
         reb_rate,
         opp_true_fg_pct,
         off_rating,
         def_rating,
         net_rating,
         pace) %>%
  left_join(seeds_cleaned %>% 
              select(TeamID, Season, top_seeded_teams) %>%
              rename(LTeamID = TeamID),
            by = c("LTeamID", "Season")) %>%
  mutate(top_seeded_teams = ifelse(is.na(top_seeded_teams),
                                   0,
                                   top_seeded_teams)) %>%
  select(-LTeamID) %>%
  bind_rows(rs_results_plus %>% 
              mutate(poss = LFGA + 0.475*LFTA - LOR + LTO,
                     opp_poss = WFGA + 0.475*WFTA - WOR + WTO,
                     ass_ratio = (LAst*100)/(LFGA + (LFTA*0.44) + LAst + LTO),
                     tov_ratio = (LTO*100)/(LFGA + (LFTA*0.44) + LAst + LTO),
                     reb_rate = ((LOR+LDR)*200)/(40*(WOR+WDR+LOR+LDR)),
                     opp_true_fg_pct = (WScore*50)/(WFGA + (WFTA*0.44)),
                     off_rating = round((LScore / poss)*100, 2),
                     def_rating = round((WScore / opp_poss)*100, 2), 
                     net_rating = off_rating - def_rating,
                     pace = 40*((poss+opp_poss)/(2*(200/5)))) %>%
              rename(TeamID = LTeamID,
                     Score_left = WScore,
                     Score_right = LScore,
                     Loc = WLoc) %>%
              mutate(W_bin = 0) %>%
              select(TeamID,
                     WTeamID,
                     Season, 
                     DayNum,
                     Score_left,
                     Score_right,
                     Loc,
                     W_bin,
                     poss,
                     opp_poss,
                     ass_ratio,
                     tov_ratio,
                     reb_rate,
                     opp_true_fg_pct,
                     off_rating,
                     def_rating,
                     net_rating,
                     pace) %>%
              left_join(seeds_cleaned %>% 
                          select(TeamID, Season, top_seeded_teams) %>%
                          rename(WTeamID = TeamID),
                        by = c("WTeamID", "Season")) %>%
              mutate(top_seeded_teams = ifelse(is.na(top_seeded_teams),
                                               0,
                                               top_seeded_teams)) %>%
              select(-WTeamID)) 

df_teams_plus = df_games %>%
  group_by(TeamID, Season) %>%
  summarise(G = n(),
            G_vs_topseeds = sum(top_seeded_teams == 1),
            W = sum(W_bin == 1),
            L = sum(W_bin == 0),
            W_last30D = sum(W_bin == 1 & DayNum > 100),
            L_last30D = sum(W_bin == 0 & DayNum > 100),
            W_H = sum(W_bin == 1 & Loc == "H"),
            W_A = sum(W_bin == 1 & Loc == "A"),
            W_N = sum(W_bin == 1 & Loc == "N"),
            W_vs_topseeds = 
              sum(W_bin == 1 & top_seeded_teams == 1),
            PS = median(Score_left),
            PS_H = median(Score_left[Loc == "H"]),
            PS_A = median(Score_left[Loc == "A"]),
            PS_N = median(Score_left[Loc == "N"]),
            PS_last30D = median(Score_left[DayNum > 100]),
            PA = median(Score_right),
            PA_H = median(Score_right[Loc == "H"]),
            PA_A = median(Score_right[Loc == "A"]),
            PA_N = median(Score_right[Loc == "N"]),
            PA_last30D = median(Score_right[DayNum > 100]),
            poss_m = median(poss),
            opp_poss_m = median(opp_poss),
            ass_ratio_m = median(ass_ratio),
            tov_ratio_m = median(tov_ratio),
            reb_rate_m = median(reb_rate),
            opp_true_fg_pct_m = median(opp_true_fg_pct),
            off_rating_m = median(off_rating),
            def_rating_m = median(def_rating),
            net_rating_m = median(net_rating),
            pace_m = median(pace),
            off_rating_m_last30D = median(off_rating[DayNum > 100]),
            def_rating_m_last30D = median(def_rating[DayNum > 100]),
            net_rating_m_last30D = median(net_rating[DayNum > 100]),
            off_rating_m_vs_topseeds = 
              median(off_rating[top_seeded_teams == 1]),
            def_rating_m_vs_topseeds = 
              median(def_rating[top_seeded_teams == 1]),
            net_rating_m_vs_topseeds = 
              median(net_rating[top_seeded_teams == 1]),
            poss_m_vs_topseeds = 
              median(poss[top_seeded_teams == 1]),
            opp_poss_m_vs_topseeds = 
              median(opp_poss[top_seeded_teams == 1]),
            ass_ratio_m_vs_topseeds = 
              median(ass_ratio[top_seeded_teams == 1]),
            tov_ratio_m_vs_topseeds = 
              median(tov_ratio[top_seeded_teams == 1]),
            reb_rate_m_vs_topseeds = 
              median(reb_rate[top_seeded_teams == 1]),
            opp_true_fg_pct_m_vs_topseeds = 
              median(opp_true_fg_pct[top_seeded_teams == 1])) %>%
  ungroup() %>%
  mutate(G_last30D = W_last30D + L_last30D,
         W_PCT = round(W / G, 2),
         L_PCT = 1 - W_PCT,
         W_PCT_last30D = round(W_last30D / G_last30D, 2),
         L_PCT_last30D = 1 - W_PCT_last30D,
         W_PCT_vs_topseeds = round(W_vs_topseeds / G_vs_topseeds, 2),
         L_PCT_vs_topseeds = 1 - W_PCT_vs_topseeds) %>%
  left_join(teams %>% 
              select(TeamID, TeamName), 
            by = "TeamID") %>%
  left_join(seeds_cleaned %>% 
              select(-top_seeded_teams),
            by = c("TeamID", "Season"))

# clean coach information
coach_score = coaches %>%
  left_join(mm_results %>%
              filter(DayNum == 154) %>%
              select(TeamID = WTeamID, Season) %>%
              mutate(ncaa_champ = 1), 
            by = c("Season", "TeamID")) %>%
  left_join(df_teams_plus %>%
              select(TeamID, Season, 
                     W_PCT, L_PCT,
                     W_PCT_vs_topseeds, 
                     L_PCT_vs_topseeds), 
            by = c("Season", "TeamID")) %>%
  group_by(CoachName) %>%
  summarise(c_N_season = n(),
            c_N_champ_W = sum(ncaa_champ == 1, 
                              na.rm = T),
            c_W_PCT_allT = mean(W_PCT, na.rm = T),
            c_W_PCT_allT = ifelse(is.na(c_W_PCT_allT), 0.3, c_W_PCT_allT),
            c_W_PCT_vs_topseeds_allT = mean(W_PCT_vs_topseeds, na.rm = T),
            c_W_PCT_vs_topseeds_allT = 
              ifelse(is.na(c_W_PCT_vs_topseeds_allT), 
                     0.3, c_W_PCT_vs_topseeds_allT),
            c_W_PCT_vs_topseeds_allT_lastyears = 
              mean(W_PCT_vs_topseeds[Season %in% 2010:2017], na.rm = T),
            c_W_PCT_vs_topseeds_allT_lastyears = 
              ifelse(is.na(c_W_PCT_vs_topseeds_allT_lastyears), 
                     0.3, c_W_PCT_vs_topseeds_allT_lastyears)) %>%
  arrange(desc(c_N_season)) %>%
  mutate(nrm_c_wins = 
           (c_W_PCT_allT - min(c_W_PCT_allT, na.rm = T)) / 
           (max(c_W_PCT_allT, na.rm = T) - min(c_W_PCT_allT, na.rm = T)),
         nrm_c_top_wins = 
           (c_W_PCT_vs_topseeds_allT - 
              min(c_W_PCT_vs_topseeds_allT, na.rm = T)) / 
           (max(c_W_PCT_vs_topseeds_allT, na.rm = T) - 
              min(c_W_PCT_vs_topseeds_allT, na.rm = T)),
         nrm_c_season = 
           (c_N_season - min(c_N_season, na.rm = T)) / 
           (max(c_N_season, na.rm = T) - min(c_N_season, na.rm = T)),
         nrm_c_champ_wins = 
           (c_N_champ_W - min(c_N_champ_W, na.rm = T)) / 
           (max(c_N_champ_W, na.rm = T) - min(c_N_champ_W, na.rm = T))) 

coach_cleaned = coaches %>%
  select(CoachName, TeamID, Season) %>%
  left_join(coach_score %>%
              select(CoachName, nrm_c_wins, 
                     nrm_c_top_wins, nrm_c_season, 
                     nrm_c_champ_wins),
            by = "CoachName") %>%
  group_by(TeamID, Season) %>%
  mutate(N = row_number()) %>%
  filter(N == max(N))

massey_clean = massey %>%
  mutate(RankingDayNum = ifelse(Season == 2018 & RankingDayNum == 128, 
                                133, RankingDayNum)) %>%
  filter(SystemName %in% c("POM", "MOR", "SAG"),
         Season %in% 2008:2018,
         RankingDayNum == 133) %>%
  spread(SystemName, OrdinalRank) %>%
  mutate_at(vars(MOR, POM, SAG), funs(ifelse(is.na(.), max(., na.rm = T), .))) %>%
  as.data.frame()

massey_clean = massey_clean %>%
  group_by(Season) %>%
  mutate(nrm_MOR = 1 - ((MOR - min(MOR)) / (max(MOR) - min(MOR))),
         nrm_POM = 1 - ((POM - min(POM)) / (max(POM) - min(POM))),
         nrm_SAG = 1- ((SAG - min(SAG)) / (max(SAG) - min(SAG)))) %>%
  select(-RankingDayNum) 

df_teams_plus = df_teams_plus %>%
  left_join(coach_cleaned %>%
              select(TeamID, Season, nrm_c_wins, 
                     nrm_c_top_wins, nrm_c_season, 
                     nrm_c_champ_wins),
            by = c("TeamID", "Season")) %>%
  left_join(massey_clean %>%
              select(TeamID, Season, nrm_MOR,
                     nrm_POM,nrm_SAG),
            by = c("TeamID", "Season")) 


# create a normalized score per season
df_teams_plus_final = df_teams_plus

# selecting from 2008 to 2013 in order to have historical data 
# without considering too much old results
mm_200817 = mm_results %>% 
  filter(Season %in% 2008:2017)

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
training = create_df_training(mm_200817, df_Xs)

# imputing missing values with the mean
training = training %>%
  mutate_all(funs(ifelse(is.na(.), 
                         mean(., na.rm = T),
                         .)))
save(training, 
     file = "datasets/10R_dataset_training_2008_2017.rdata")

save(df_teams_plus_final, 
     file = "datasets/10R_teams_with_advanced_metrics.rdata")
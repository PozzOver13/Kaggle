import re
import numpy as np 
import pandas as pd 

def clean_seed(seed):
    s_int = int(seed[1:3])
    return s_int

def extract_seed_region(seed):
    s_reg = seed[0:1]
    return s_reg

def new_name_w_1(old_name):
    match = re.match(r'^L', old_name)
    if match:
        out = re.sub('^L','', old_name)
        return out + '_opp'
    return old_name

def new_name_w_2(old_name):
    match = re.match(r'^W', old_name)
    if match:
        out = re.sub('^W','', old_name)
        return out
    return old_name

def prepare_stats_extended_winners(df_in, df_seed_in, df_teams_in):
    df_in['poss'] = df_in['WFGA'] + 0.475*df_in['WFTA'] - df_in['WOR'] + df_in['WTO']
    df_in['opp_poss'] = df_in['LFGA'] + 0.475*df_in['LFTA'] - df_in['LOR'] + df_in['LTO']
    df_in['ass_ratio'] = (df_in['WAst']*100)/(df_in['WFGA'] + (df_in['WFTA']*0.44) + df_in['WAst'] + df_in['WTO'])
    df_in['tov_ratio'] = (df_in['WTO']*100)/(df_in['WFGA'] + (df_in['WFTA']*0.44) + df_in['WAst'] + df_in['WTO'])
    df_in['reb_rate'] = ((df_in['WOR']+df_in['WDR'])*200)/(40*(df_in['WOR']+ df_in['WDR']+ df_in['LOR']+ df_in['LDR']))
    df_in['opp_true_fg_pct'] = (df_in['LScore']*50)/(df_in['LFGA'] + (df_in['LFTA']*0.44))
    df_in['off_rating'] = 100*(df_in['WScore'] / df_in['poss'])
    df_in['def_rating'] = 100*(df_in['LScore'] / df_in['opp_poss'])
    df_in['net_rating'] = df_in['off_rating'] - df_in['def_rating']
    df_in['pace'] = 48*((df_in['poss']+df_in['opp_poss'])/(2*(240/5)))
    
    df_in = df_in.rename(columns={'WTeamID':'TeamID', 
                                  'WLoc':'_Loc',
                                  'LTeamID':'TeamID_opp',
                                  'WScore':'Score_left', 
                                  'LScore':'Score_right'})
    
    df_seeds_opp = df_seed_in.rename(columns={'TeamID':'TeamID_opp',
                                              'seed_int':'seed_int_opp',
                                              'seed_region':'seed_region_opp',
                                              'top_seeded_teams':'top_seeded_teams_opp'})
    
    df_out = pd.merge(left=df_in, right=df_seed_in, how='left', on=['Season', 'TeamID'])
    df_out = pd.merge(left=df_out, right=df_seeds_opp, how='left', on=['Season', 'TeamID_opp'])
    df_out = pd.merge(left=df_out, right=df_teams_in, how='left', on=['TeamID'])
    
    df_out['DayNum'] = pd.to_numeric(df_out['DayNum'])
    df_out['win_dummy'] = 1
    
    df_out['seed_int'] = np.where(df_out['seed_int'].isnull(), 20, df_out['seed_int'])
    df_out['seed_region'] = np.where(df_out['seed_region'].isnull(), 'NoTour', df_out['seed_region'])
    df_out['top_seeded_teams'] = np.where(df_out['top_seeded_teams'].isnull(), 0, df_out['top_seeded_teams'])
    
    df_out['seed_int_opp'] = np.where(df_out['seed_int_opp'].isnull(), 20, df_out['seed_int_opp'])
    df_out['seed_region_opp'] = np.where(df_out['seed_region_opp'].isnull(), 'NoTour', df_out['seed_region_opp'])
    df_out['top_seeded_teams_opp'] = np.where(df_out['top_seeded_teams_opp'].isnull(), 0, df_out['top_seeded_teams_opp'])
    
    df_out = df_out.rename(columns=new_name_w_1)
    df_out = df_out.rename(columns=new_name_w_2)
    
    return df_out


def new_name_l_1(old_name):
    match = re.match(r'^W', old_name)
    if match:
        out = re.sub('^W','', old_name)
        return out + '_opp'
    return old_name

def new_name_l_2(old_name):
    match = re.match(r'^L', old_name)
    if match:
        out = re.sub('^L','', old_name)
        return out
    return old_name

def prepare_stats_extended_losers(df_in, df_seed_in, df_teams_in):
    df_in['poss'] = df_in['LFGA'] + (0.475*df_in['LFTA']) - df_in['LOR'] + df_in['LTO']
    df_in['opp_poss'] = df_in['WFGA'] + (0.475*df_in['WFTA']) - df_in['WOR'] + df_in['WTO']
    df_in['ass_ratio'] = (df_in['LAst']*100)/(df_in['LFGA'] + (df_in['LFTA']*0.44) + df_in['LAst'] + df_in['LTO'])
    df_in['tov_ratio'] = (df_in['LTO']*100)/(df_in['LFGA'] + (df_in['LFTA']*0.44) + df_in['LAst'] + df_in['LTO'])
    df_in['reb_rate'] = ((df_in['LOR']+df_in['LDR'])*200)/(40*(df_in['WOR']+ df_in['WDR']+ df_in['LOR']+ df_in['LDR']))
    df_in['opp_true_fg_pct'] = (df_in['WScore']*50)/(df_in['WFGA'] + (df_in['WFTA']*0.44))
    df_in['off_rating'] = 100*(df_in['LScore'] / df_in['poss'])
    df_in['def_rating'] = 100*(df_in['WScore'] / df_in['opp_poss'])
    df_in['net_rating'] = df_in['off_rating'] - df_in['def_rating']
    df_in['pace'] = 48*((df_in['poss']+df_in['opp_poss'])/(2*(240/5)))
    
    df_in = df_in.rename(columns={'LTeamID':'TeamID', 
                                  'LLoc':'_Loc',
                                  'WTeamID':'TeamID_opp',
                                  'LScore':'Score_left', 
                                  'WScore':'Score_right'})
    
    df_seeds_opp = df_seed_in.rename(columns={'TeamID':'TeamID_opp',
                                              'seed_int':'seed_int_opp',
                                              'seed_region':'seed_region_opp',
                                              'top_seeded_teams':'top_seeded_teams_opp'})
    
    df_out = pd.merge(left=df_in, right=df_seed_in, how='left', on=['Season', 'TeamID'])
    df_out = pd.merge(left=df_out, right=df_seeds_opp, how='left', on=['Season', 'TeamID_opp'])
    df_out = pd.merge(left=df_out, right=df_teams_in, how='left', on=['TeamID'])
    
    df_out['DayNum'] = pd.to_numeric(df_out['DayNum'])
    df_out['win_dummy'] = 0
    
    df_out['seed_int'] = np.where(df_out['seed_int'].isnull(), 20, df_out['seed_int'])
    df_out['seed_region'] = np.where(df_out['seed_region'].isnull(), 'NoTour', df_out['seed_region'])
    df_out['top_seeded_teams'] = np.where(df_out['top_seeded_teams'].isnull(), 0, df_out['top_seeded_teams'])
    
    df_out['seed_int_opp'] = np.where(df_out['seed_int_opp'].isnull(), 20, df_out['seed_int_opp'])
    df_out['seed_region_opp'] = np.where(df_out['seed_region_opp'].isnull(), 'NoTour', df_out['seed_region_opp'])
    df_out['top_seeded_teams_opp'] = np.where(df_out['top_seeded_teams_opp'].isnull(), 0, df_out['top_seeded_teams_opp'])

    df_out = df_out.rename(columns=new_name_l_1)
    df_out = df_out.rename(columns=new_name_l_2)
    
    return df_out

def aggr_stats(df):
    d = {}
    d['G'] = df['win_dummy'].count()
    d['W'] = df['win_dummy'].sum()
    d['L'] = np.sum(df['win_dummy'] == 0)
    d['G_vs_topseeds'] = np.sum(df['top_seeded_teams_opp'] == 1)
    d['W_vs_topseeds'] = np.sum((df['win_dummy'] == 1) & (df['top_seeded_teams_opp'] == 1))
    d['L_vs_topseeds'] = np.sum((df['win_dummy'] == 0) & (df['top_seeded_teams_opp'] == 1))
    d['G_last30D'] = np.sum((df['DayNum'] > 100))
    d['W_last30D'] = np.sum((df['win_dummy'] == 1) & (df['DayNum'] > 100))
    d['L_last30D'] = np.sum((df['win_dummy'] == 0) & (df['DayNum'] > 100))

    d['G_first30D'] = np.sum((df['DayNum'] <= 100))
    d['W_first30D'] = np.sum((df['win_dummy'] == 1) & (df['DayNum'] <= 100))
    d['L_first30D'] = np.sum((df['win_dummy'] == 0) & (df['DayNum'] <= 100))
	
    d['G_H'] = np.sum((df['_Loc'] == 'H'))
    d['W_H'] = np.sum((df['win_dummy'] == 1) & (df['_Loc'] == 'H'))
    d['L_H'] = np.sum((df['win_dummy'] == 0) & (df['_Loc'] == 'H'))
    d['G_A'] = np.sum((df['_Loc'] == 'A'))
    d['W_A'] = np.sum((df['win_dummy'] == 1) & (df['_Loc'] == 'A'))
    d['L_A'] = np.sum((df['win_dummy'] == 0) & (df['_Loc'] == 'A'))
    d['G_N'] = np.sum((df['_Loc'] == 'N'))
    d['W_N'] = np.sum((df['win_dummy'] == 1) & (df['_Loc'] == 'N'))
    d['L_N'] = np.sum((df['win_dummy'] == 0) & (df['_Loc'] == 'N'))
    
    d['PS'] = np.mean(df['Score_left'])
    d['PS_H'] = np.mean(df['Score_left'][df['_Loc'] == 'H'])
    d['PS_A'] = np.mean(df['Score_left'][df['_Loc'] == 'A'])
    d['PS_N'] = np.mean(df['Score_left'][df['_Loc'] == 'N'])
    d['PS_last30D'] = np.mean(df['Score_left'][df['DayNum'] > 100])
    
    d['PA'] = np.mean(df['Score_right'])
    d['PA_H'] = np.mean(df['Score_right'][df['_Loc'] == 'H'])
    d['PA_A'] = np.mean(df['Score_right'][df['_Loc'] == 'A'])
    d['PA_N'] = np.mean(df['Score_right'][df['_Loc'] == 'N'])
    d['PA_last30D'] = np.mean(df['Score_right'][df['DayNum'] > 100])
    
    d['poss_m'] = np.mean(df['poss'][df['top_seeded_teams_opp'] == 1])
    d['opp_poss_m'] = np.mean(df['opp_poss'][df['top_seeded_teams_opp'] == 1])
    d['ass_ratio_m'] = np.mean(df['ass_ratio'][df['top_seeded_teams_opp'] == 1])
    d['tov_ratio_m'] = np.mean(df['tov_ratio'][df['top_seeded_teams_opp'] == 1])
    d['reb_rate_m'] = np.mean(df['reb_rate'][df['top_seeded_teams_opp'] == 1])
    d['opp_true_fg_pct_m'] = np.mean(df['opp_true_fg_pct'][df['top_seeded_teams_opp'] == 1])
    d['off_rating_m'] = np.mean(df['off_rating'][df['top_seeded_teams_opp'] == 1])
    d['def_rating_m'] = np.mean(df['def_rating'][df['top_seeded_teams_opp'] == 1])
    d['net_rating_m'] = np.mean(df['net_rating'][df['top_seeded_teams_opp'] == 1])
    d['pace_m'] = np.mean(df['pace'][df['top_seeded_teams_opp'] == 1])
    
    d['poss_s'] = np.std(df['poss'][df['top_seeded_teams_opp'] == 1])
    d['opp_poss_s'] = np.std(df['opp_poss'][df['top_seeded_teams_opp'] == 1])
    d['ass_ratio_s'] = np.std(df['ass_ratio'][df['top_seeded_teams_opp'] == 1])
    d['tov_ratio_s'] = np.std(df['tov_ratio'][df['top_seeded_teams_opp'] == 1])
    d['reb_rate_s'] = np.std(df['reb_rate'][df['top_seeded_teams_opp'] == 1])
    d['opp_true_fg_pct_s'] = np.std(df['opp_true_fg_pct'][df['top_seeded_teams_opp'] == 1])
    d['off_rating_s'] = np.std(df['off_rating'][df['top_seeded_teams_opp'] == 1])
    d['def_rating_s'] = np.std(df['def_rating'][df['top_seeded_teams_opp'] == 1])
    d['net_rating_s'] = np.std(df['net_rating'][df['top_seeded_teams_opp'] == 1])
    d['pace_s'] = np.std(df['pace'][df['top_seeded_teams_opp'] == 1])
    
    d['off_rating_m_last30D'] = np.mean(df['off_rating'][df['DayNum'] > 100])
    d['def_rating_m_last30D'] = np.mean(df['def_rating'][df['DayNum'] > 100])
    d['net_rating_m_last30D'] = np.mean(df['net_rating'][df['DayNum'] > 100])
    
    d['off_rating_m_vs_topseeds'] = np.mean(df['off_rating'][df['top_seeded_teams_opp'] == 1])
    d['def_rating_m_vs_topseeds'] = np.mean(df['def_rating'][df['top_seeded_teams_opp'] == 1])
    d['net_rating_m_vs_topseeds'] = np.mean(df['net_rating'][df['top_seeded_teams_opp'] == 1])
    
    return pd.Series(d)


def new_name_tourn(str_in, col_list_in, string_out = 'left'):
    out_name = {}
    for old_name in str_in:
        match = old_name in col_list_in
        if match:
            out_name[old_name] = old_name + '_' + string_out
    
    return out_name

def prepare_tournament_datasets(df_tourn_in, df_agg_stats_in, 
                                df_coach_in, df_massey_in,
                                list_feat_in):
    
    df_tourn_in['TeamID'] = df_tourn_in[['WTeamID','LTeamID']].min(axis=1)
    df_tourn_in['TeamID_opp'] = df_tourn_in[['WTeamID','LTeamID']].max(axis=1)
    df_tourn_in['win_dummy'] = np.where(df_tourn_in['TeamID'] == df_tourn_in['WTeamID'], 1, 0)
    df_tourn_in['delta'] = np.where(df_tourn_in['win_dummy'] == 1,
                                    df_tourn_in['WScore'] - df_tourn_in['LScore'],
                                    df_tourn_in['LScore'] - df_tourn_in['WScore'])
    df_tourn_in['Score_left'] = np.where(df_tourn_in['win_dummy'] == 1,
                                         df_tourn_in['WScore'],
                                         df_tourn_in['LScore'])
    df_tourn_in['Score_right'] = np.where(df_tourn_in['win_dummy'] == 1,
                                          df_tourn_in['LScore'],
                                          df_tourn_in['WScore'])
        
    # aggregate stats
    names_l = new_name_tourn(list_feat_in, list_feat_in, 'left')
    df_teams_gr_left = df_agg_stats_in.loc[:,['Season', 'TeamID'] + list_feat_in].\
                  rename(columns=names_l)
    
    names_r = new_name_tourn(list_feat_in, list_feat_in, 'right')
    names_r['TeamID'] = 'TeamID_opp'
    df_teams_gr_right = df_agg_stats_in.loc[:,['Season', 'TeamID'] + list_feat_in].\
                  rename(columns=names_r)
    
    # coach
    regr_c = ['CoachName', 'c_N_season', 'c_N_champ_W', 'c_W_PCT_allT', 'c_W_PCT_vs_topseeds_allT']
    names_c_l = new_name_tourn(regr_c, regr_c, 'left')
    names_c_r = new_name_tourn(regr_c, regr_c, 'right')
    names_c_r['TeamID'] = 'TeamID_opp'
    
    df_coach_left = df_coach_in.rename(columns=names_c_l)
    df_coach_right = df_coach_in.rename(columns=names_c_r)
    
    # massey
    regr_m = ['MOR', 'POM', 'SAG']
    names_m_l = new_name_tourn(regr_m, regr_m, 'left')
    names_m_r = new_name_tourn(regr_m, regr_m, 'right')
    names_m_r['TeamID'] = 'TeamID_opp'

    df_massey_left = df_massey_in.rename(columns=names_m_l)
    df_massey_right = df_massey_in.rename(columns=names_m_r)
    
    df_tourn_out = pd.merge(left=df_tourn_in, 
                            right=df_teams_gr_left, 
                            how='left', on=['Season', 'TeamID'])
    df_tourn_out = pd.merge(left=df_tourn_out, 
                            right=df_teams_gr_right, 
                            how='left', on=['Season', 'TeamID_opp'])
    df_tourn_out = pd.merge(left=df_tourn_out, 
                            right=df_coach_left, 
                            how='left', on=['Season', 'TeamID'])
    df_tourn_out = pd.merge(left=df_tourn_out, 
                            right=df_coach_right, 
                            how='left', on=['Season', 'TeamID_opp'])
    df_tourn_out = pd.merge(left=df_tourn_out, 
                            right=df_massey_left, 
                            how='left', on=['Season', 'TeamID'])
    df_tourn_out = pd.merge(left=df_tourn_out, 
                            right=df_massey_right, 
                            how='left', on=['Season', 'TeamID_opp'])
    
    
    # create delta values
    regr_c.remove('CoachName')
    
    delta_vars = list_feat_in + regr_c
    delta_vars = delta_vars + regr_m
    for var in delta_vars:
        
        df_tourn_out['delta_' + var] = df_tourn_out[var + '_left'] - df_tourn_out[var + '_right']
    
    df_out = df_tourn_out.loc[:, ['Season', 'DayNum',
                                  'TeamID', 'TeamID_opp',
                                  'Score_left', 'Score_right',
                                  'win_dummy', 
                                  'delta', 'NumOT'] + ['delta_' + s for s in delta_vars]]
                                    
    return df_out



def somers2_py(x, y):
    
    from sklearn.metrics import roc_auc_score
    
    C = roc_auc_score(y, x)
    Dxy = (2 * roc_auc_score(y, x))  - 1
    
    return Dxy, C

def apply_somers(df):
    
    d = {}
    
    dxy, cxy = somers2_py(df['value'],
                          df['win_dummy'])
    
    d['Dxy'] = dxy
    d['C'] = cxy
    return pd.Series(d)
	

def aggr_stats_coach(df):
    d = {}
    d['c_N_season'] = df['ncaa_champ'].count()
    d['c_N_champ_W'] = df['ncaa_champ'].sum()
    d['c_W_PCT_allT'] = np.mean(df['w_pct'][df['Season'] >= 2003])
    d['c_W_PCT_vs_topseeds_allT'] = np.mean(df['w_pct_vs_topseeds'][df['Season'] >= 2003])
    
    return pd.Series(d)

def prepare_stats_coach(df_coach_in, df_tourn_in, df_stats_agg_in):
    
    df_tourn_t = df_tourn_in[df_tourn_in['DayNum'] == 154].rename(columns={'WTeamID':'TeamID'})
    df_tourn_t = df_tourn_t.loc[:,['TeamID', 'Season']]
    df_tourn_t['ncaa_champ'] = 1
    
    coaches_cl = pd.merge(left=df_coach_in.copy(), 
                          right=df_tourn_t.copy(), 
                          how='left', on=['Season', 'TeamID'])
    df_agg_t = df_stats_agg_in.copy()
    
    coaches_cl = pd.merge(left=coaches_cl.copy(), 
                          right=df_agg_t.loc[:, ['TeamID', 'Season', 'G',
                                                 'w_pct', 'w_pct_last30D','w_pct_vs_topseeds']], 
                          how='left', on=['Season', 'TeamID'])
    
    coaches_cl['ncaa_champ'] = np.where(coaches_cl['ncaa_champ'].isnull(), 0, 1)
    coaches_cl['w_pct'] = np.where(coaches_cl['w_pct'].isnull(), 0, coaches_cl['w_pct'])
    coaches_cl['w_pct_last30D'] = np.where(coaches_cl['w_pct_last30D'].isnull(), 0, coaches_cl['w_pct_last30D'])
    coaches_cl['w_pct_vs_topseeds'] = np.where(coaches_cl['w_pct_vs_topseeds'].isnull(), 0, coaches_cl['w_pct_vs_topseeds'])
    coaches_cl['Season_rif'] = coaches_cl['Season']
    
    year_u = df_coach_in['Season'].unique()
    year_u = year_u[year_u >= 2003]
    
    for year in year_u:
        
        coaches_t = coaches_cl[coaches_cl['Season'] == year].loc[:, ['CoachName', 'Season']]
        coaches_cl_t = coaches_cl[(coaches_cl['Season'].isin(year_u[year_u <= year]))].reindex()
        
        # aggregating by teams and seasons
        coaches_cl_agg = coaches_cl_t.\
                          groupby([coaches_cl['CoachName']]).\
                          apply(aggr_stats_coach).\
                          reset_index()

        coaches_cl_agg['Season'] = year
        coaches_out_t = pd.merge(left=coaches_t, 
                                 right=coaches_cl_agg, 
                                 how='left', on=['CoachName', 'Season']).\
                 drop_duplicates(subset=['Season', 'CoachName'], keep='first')
        
        coaches_out = pd.merge(left=coaches_cl.loc[:, ['Season', 'TeamID','CoachName']].copy(), 
                      right=coaches_out_t, 
                      how='left', on=['CoachName', 'Season']).\
                 drop_duplicates(subset=['Season', 'TeamID'], keep='first')
        
        coaches_out = coaches_out[coaches_out['Season'] == year]
        
        if year == 2003:
            coaches_out_f = coaches_out
        else:
            coaches_out_f = pd.concat([coaches_out_f, coaches_out])
            
    return(coaches_out_f)
	
def prepare_massey_ord(massey_in):
    massey_t = massey_in.copy()

    massey_t = massey_t[(massey_t['SystemName'].isin(['POM', 'MOR', 'SAG'])) & \
                        (massey_t['Season'].isin(np.arange(2008, 2020))) & \
                        (massey_t['RankingDayNum'] > 130)]
    
    massey_out = massey_t.drop(['RankingDayNum'], axis=1).\
             pivot_table(index=['TeamID', 'Season'], 
                         columns='SystemName', 
                         values='OrdinalRank').reset_index()
    
    return(massey_out)
	
def ingest_submission(sub_in):
    
    mysub_out = sub_in.copy()
    expl_id = mysub_out['ID'].str.split(pat = "_", expand=True)
    mysub_out['Season'] = expl_id.loc[:,0].astype(int)
    mysub_out['DayNum'] = 1
    mysub_out['WTeamID'] = expl_id.loc[:,1].astype(int)
    mysub_out['WScore'] = 0
    mysub_out['LTeamID'] = expl_id.loc[:,2].astype(int)
    mysub_out['LScore'] = 0
    mysub_out['WLoc'] = '' 
    mysub_out['NumOT'] = 0
    mysub_out = mysub_out.drop(columns=['ID', 'Pred'])
    
    return(mysub_out)
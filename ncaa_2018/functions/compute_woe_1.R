# function to compute multiple woe transformation
compute_woe <- function(df_in, var, n_class = 8) {
  res_woe <- woe(Data = df_in, Independent = var, 
                 Continuous = TRUE, Dependent = "target", 
                 C_Bin = n_class, Bad = 0, Good = 1)
  
  control = sum(table(c(res_woe$MIN, max(res_woe$MAX))) > 1)
  
  if (control == 0) {
    return(res_woe)
  } else {
    while(!(control == 0)) {
      res_woe_new <- woe(Data = df, Independent = var, 
                         Continuous = TRUE, Dependent = "target", 
                         C_Bin = (n_class - 1), Bad = 0, Good = 1)
      control = sum(table(c(res_woe_new$MIN, max(res_woe_new$MAX))) > 1)
      n_class = n_class - 1
    }
    return(res_woe_new)
  }
}
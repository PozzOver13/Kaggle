
# my personal ggplot theme
# rk = to be developed
theme_pozzover <-  theme(
  panel.grid.major.y = element_line(colour = "gray50", 
                                    size = 0.5,
                                    linetype = 2),
  panel.grid.minor.y = element_blank(),
  panel.background = element_rect(fill = "white", 
                                  color = "gray50"),
  axis.title = element_text(colour = "midnightblue",
                           family = "Helvetica",
                           size = 12,
                           face = "italic"),
  axis.text = element_text(colour = "midnightblue",
                           family = "Helvetica",
                           face = "italic"),
  plot.title = element_text(colour = "midnightblue",
                            family = "Helvetica",
                            face = "bold"),
  legend.title = element_text(colour = "midnightblue",
                              family = "Helvetica",
                              face = "bold")
        )

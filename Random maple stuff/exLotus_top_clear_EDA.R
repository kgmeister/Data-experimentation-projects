setwd("C:/exLotus_EDA")
library(tidyverse)
ranking_final<- read.csv('exLotus_ranking_final.csv')


summary(ranking_final$Level)
is.numeric(ranking_final$Fd_advantage)

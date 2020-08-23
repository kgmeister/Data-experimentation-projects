setwd("C:/Programs and stuff/R Projects/Average Daily Polyclinic Attendances for Selected Diseases")
library(dplyr)
library(ggplot2)
library(tidyverse)
library(RColorBrewer)

overall<- read.csv("average-daily-polyclinic-attendances-for-selected-diseases.csv")

#-------------Splitting the table into the 5 different diseases--------------#

# urti <- filter(overall, disease=='Acute Upper Respiratory Tract infections')
# conjunctivitis <- filter(overall, disease=='Acute Conjunctivitis')
# diarrhoea <- filter(overall, disease=='Acute Diarrhoea')
# chickenpox <- filter(overall, disease=='Chickenpox')
# hfmd <- filter(overall, disease=='HFMD')


#-------------------Plotting the graph, each line separate ---------------------------#
# ggplot() + geom_line(data=urti, aes(x= factor(epi_week), y= no._of_cases, group = 1), colour = 'green') +
#    geom_line(data=conjunctivitis, aes(x= factor(epi_week), y= no._of_cases, group = 1), colour = 'red') +
#    geom_line(data=diarrhoea, aes(x= factor(epi_week), y= no._of_cases, group = 1), colour = 'black') +
#    geom_line(data=chickenpox, aes(x= factor(epi_week), y= no._of_cases, group = 1), colour = 'purple') +
#    geom_line(data=hfmd, aes(x= factor(epi_week), y= no._of_cases, group = 1), colour = 'blue')


#------------------Or plotting it as facet, with scaling. Best visualisation! ---------------------------#
plot <- ggplot() + 
  geom_line(data=overall, aes(x= factor(epi_week), y= no._of_cases, group = 1, color = no._of_cases)) +
  scale_color_gradientn(colours = rainbow(8))

plot + facet_grid(disease ~ ., scales= 'free')

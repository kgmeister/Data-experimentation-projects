getwd()
setwd("C:/Programs and stuff/R Projects/Google_play_store")
library(tidyverse)
library(ggplot2)
library(anytime)
library(dplyr)
library(stringr)

playstore<- read.csv("googleplaystore.csv")

#View(playstore)


options(scipen=999)
#------------- Changing to readable date format -------------------------

playstore$Last.Updated <-anydate(playstore$Last.Updated) 

#------------- Removing all "Varies with device" -----------------------------

playstore<- playstore[!grepl('Varies with device',playstore$Size),]

playstore$Size<- as.numeric(playstore$Size) #Because it will throw up "Discrete value supplied to continuous scale" otherwise during bubble plot


#-------------- Changing the millions into numbers and removing the "+" in installs ----------------

playstore$Size <- gsub('B', 'e9', playstore$Size)
playstore$Size <- gsub('M', 'e6', playstore$Size)
playstore$Size <- gsub('k', 'e3', playstore$Size)
playstore$Size<- format(as.numeric(playstore$Size), scientific = FALSE)


playstore$Installs <- stringr::str_replace(playstore$Installs,'\\+', '')
playstore$Installs <- as.numeric(gsub(",","",playstore$Installs), scientific = FALSE)

#-------------- Removing NAs ---------------------------
playstore<-na.omit(playstore)

#write.csv(playstore, "playstore.csv")


#--------- Bubble plot -----------------------
library(extrafontdb)
library(viridis)
library(hrbrthemes)
library(extrafont)


ggplot(data=playstore, aes(x=Rating, y= Installs, size=Size, color=Category)) + 
  geom_point(alpha=0.3) + 
  scale_fill_viridis(discrete=TRUE, guide=FALSE, option="A") +
  scale_size(range = c(0.1, 24)) +
  theme_ipsum()


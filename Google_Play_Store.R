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
###################  End of dataset cleaning ################################

ggplot(data=playstore, aes(x=Rating, y= Installs, size=Size, color=Category)) + 
  geom_point(alpha=0.3) + 
  scale_fill_viridis(discrete=TRUE, guide=FALSE, option="A") +
  scale_size(range = c(0.1, 24)) +
  theme_ipsum()

############################################################################################
############################################################################################

#--------------------- Obtaining average rating vs category ----------------------------

ratingavg <- playstore[,c('Category',"Rating")]
ratingavg <- aggregate(Rating ~ Category, ratingavg, mean)

ratingavg <- ratingavg[order(ratingavg$Rating, decreasing = TRUE),] #Sorting by decreasing Rating

#------- Bar plot of rating avg --------
ggplot(data=ratingavg, aes(x= Category, y= Rating)) +
  geom_bar(stat = "identity", fill = rainbow(n=length(ratingavg$Rating)))

#------------- taking only top 10 -----------

toprating <- ratingavg[1:10,]
View(toprating)

###########################################################################################

#------------- Bubble plot top 10 ------------------------

top10 <- playstore[playstore$Category %in% toprating$Category,]
top10$Category <- as.factor(top10$Category)

#--- Extracting out relevant variables for plot, and obtaining the average -----
top10plot <- top10[,c('Category','Rating','Size','Installs')]
top10plot <- aggregate(. ~ Category, top10plot, mean)


ggplot(data=top10plot, aes(x=Rating, y= Installs, size=Size, color=Category)) + 
  geom_point(alpha=0.8) + 
  scale_fill_viridis(discrete=TRUE, guide=FALSE, option="A") +
  scale_size(range = c(0.1, 24)) +
  theme_ipsum()

#############################################################################################
#############################################################################################

#~~~~~~~~~~~~~~~~~~ Bubble plot top 10 by Installs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#--------------------- Obtaining average Installs vs category ----------------------------

installavg <- playstore[,c('Category',"Installs")]
installavg <- aggregate(Installs ~ Category, installavg, mean)

installavg <- installavg[order(installavg$Installs, decreasing = TRUE),] 


#------------- taking only top 10 installs -----------

topinstall <- installavg[1:10,]

top10install <- playstore[playstore$Category %in% topinstall$Category,]
top10install$Category <- as.factor(top10install$Category)

top10installplot <- top10install[,c('Category','Rating','Size','Installs')]
top10installplot <- aggregate(. ~ Category, top10installplot, mean)
View(top10installplot)

ggplot(data=top10installplot, aes(x=Rating, y= Installs, size=Size, color=Category)) + 
  geom_point(alpha=0.8) + 
  scale_fill_viridis(discrete=TRUE, guide=FALSE, option="A") +
  scale_size(range = c(0.1, 24)) +
  theme_ipsum()

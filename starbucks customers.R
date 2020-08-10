getwd()
setwd("C:/R_things")
data <- read.csv("data.csv")
View(data)

# install.packages("ggplot2")
# install.packages("tidyverse")
# 
# library(tidyverse)
# library(ggplot2)
# library(scales)

#Checking if there is any NA values
any(is.na(data))

unique(data[,"spendPurchase"])
unique(data[,"loyal"])

#extracting data table of location and loyalty
loyalty <- data[,c("location","loyal")]


#making a table for the number of occurences for loyalty vs location
loyaltytable<-as.data.frame(table(loyalty))



#making a table for loyal customers
loyalcustomers <- loyaltytable[4:6,c("location","Freq")]


#making a table for disloyal customers
disloyalcustomers <- loyaltytable[1:3,c("location","Freq")]

View(loyaltytable)
head(loyalcustomers)

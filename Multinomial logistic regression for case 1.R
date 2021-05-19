case1<- read.csv("Case 1.csv")


library(ggplot2)
library(tidyverse)
library(ggpubr)

#Removing patient IDs and NA removal
case1<-case1 %>% na.omit(case1) %>% select(-Patient.ID) 


#ordering the disease stages as levels
case1$ï..Disease.stage<- factor(case1$ï..Disease.stage, order = TRUE, levels = c("F0", "F1", "F2", "F3", "F4"))

#Finding the mean for bar plot
case1_mean <- aggregate(.~ï..Disease.stage,case1, mean)

#Finding the SD for bar plot, using group_by into summarise method. This works for mean too
case1_sd <- case1%>% group_by(ï..Disease.stage)%>% summarise(A=sd(Parameter.A), B=sd(Parameter.B),
                                                 C=sd(Parameter.C), D=sd(Parameter.D), E=sd(Parameter.E))

#Primary plotting
plot1 <- ggplot() +
  geom_bar(data=case1_mean, aes(x=ï..Disease.stage, y=Parameter.D), stat="identity", fill="skyblue", alpha=0.7) +
  geom_point(data=case1, aes(x=ï..Disease.stage, y=Parameter.D)) +
  geom_errorbar(data=case1_sd, aes(x=ï..Disease.stage, ymin = case1_mean$Parameter.D - D, ymax = case1_mean$Parameter.D + D,
                                   width=0.4, colour="orange", alpha=1, size=0.01)) +
  ggtitle("Plot of Parameter D against Disease stage") + xlab("Disease stage") + ylab("Mean")

plot1

#################################################################################################################
#################################################################################################################

## Since it's a continuous variable predicting 5 categorical variables, I'll pick multinomial logistic regression
library(nnet) #main multinomial logistic regression package
library(AER) #Wald Z-tests to find p-value for multinom
library(afex)
library(car)

#Building training and test set for model
train <- sample_frac(case1, 0.7)
test <- case1

# relevel() not needed as factors are already ordered. "-1" because I don't think intercept is needed
case1model_1 <- multinom(ï..Disease.stage ~ .-1, data = train)
summary(case1model_1)


#############
z <- summary(case1model_1)$coefficients/summary(case1model_1)$standard.errors
# 2-tailed Wald z tests to test significance of coefficients for p-values
p <- (1 - pnorm(abs(z), 0, 1)) * 2
p

#coeftest function for Wald z-tests
coeftest_result_1 <- coeftest(case1model_1) #to get p-value
head(coeftest_result_1)


#Say we want just the correlations for F1. Then we can just extract F1's parameters from the coeftest
case1model_1_F1_params <- coeftest_result_1[1:5,]

# Reordering levels so model takes F4 as reference instead to show F0 (??? to clarify with team)
train2 <- train
train2$ï..Disease.stage <- factor(train2$ï..Disease.stage, order = TRUE, levels = c("F4","F3","F2","F1","F0"))

#running model on re-ordered dataframe to get my F0 back
case1model_2 <- multinom(ï..Disease.stage ~ .-1, data = train2)
summary(case1model_2)


head(probability.table <- fitted(case1model_1))

######################################################################################
######################################################################################

## Deploying model
# Predicting the values for dataset
train$predicted <- predict(case1model_1, newdata = train, "class")

# Building classification table
ctable <- table(train$ï..Disease.stage, train$predicted)

# Calculating accuracy - sum of diagonal elements divided by total obs
round((sum(diag(ctable))/sum(ctable))*100,2)


# Using above model for test, same steps
test$predicted <- predict(case1model_1, newdata = test, "class")
ctable2 <- table(test$ï..Disease.stage, test$predicted)
round((sum(diag(ctable2))/sum(ctable2))*100,2)

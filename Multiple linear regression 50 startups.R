setwd("C:/Programs and stuff/R Projects/Udemy Stuff/Machine-Learning-Dataset/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/Multiple_Linear_Regression")

# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('50_Startups.csv')
View(dataset)

#Encoding categorical data:
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))


# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Feature Scaling
# training_set = scale(training_set[,2:3])
# test_set = scale(test_set[,2:3])


# Fitting Multiple Linear Regression to the Training set
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = training_set)

#Alternatively as a shortcut: regressor = lm(formula = Profit ~ ., data = training_set)
               
summary(regressor)

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)
y_pred


#======================   Building the optimal model using Backward Elimination  ===========================

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset) #Using whole dataset to use all the data for this, training_set is also fine
summary(regressor)


#Attempt 2
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)  
summary(regressor)


#Attempt 3
regressor = lm(formula = Profit ~ R.D.Spend + Administration,
               data = dataset)
summary(regressor)


#Attempt 4
regressor = lm(formula = Profit ~ R.D.Spend,
               data = dataset)
summary(regressor)




#========================== Automatic Backward Elimination   ===========================================

backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination(training_set, SL)


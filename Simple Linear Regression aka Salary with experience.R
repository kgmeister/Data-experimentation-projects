setwd("C:/Programs and stuff/R Projects/Udemy Stuff/Machine-Learning-Dataset/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 4 - Simple Linear Regression/Simple_Linear_Regression")

#Importing dataset
dataset <- read.csv("Salary_Data.csv")


#Splitting the dataset into Training set and Test set
library(caTools)
set.seed(123)


split <- sample.split(dataset$Salary, SplitRatio = 2/3)
split
training_set <-subset(dataset, split == TRUE)
test_set <-subset(dataset, split == FALSE)


#Feature scaling
#training_set[,2:3] <- scale(training_set[,2:3])
#test_set[,2:3] <- scale(test_set[,2:3])


#Fitting Simple Linear Regression to the Training set
regressor <- lm(formula = Salary ~ YearsExperience, 
                data = training_set)

summary(regressor)


#Predicting the TEST set results
y_pred <-predict(regressor, newdata = test_set)
y_pred


#Visualising the Training set results
library(ggplot2)

ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
            colour = "red") +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)), #plotting predicted salaries of training
            colour = "blue") +                                                                     #set observation  
  ggtitle("Salary vs Experience (Training set)") +
  xlab("Years of experience")+
  ylab("Salary")

#Visualising the Test set results
library(ggplot2)

ggplot() + 
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = "red") +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)), #plotting predicted salaries of training
            colour = "blue") +                                                                     #set observation  
  ggtitle("Salary vs Experience (Test set)") +
  xlab("Years of experience")+
  ylab("Salary")
  
                                                                                                  
            
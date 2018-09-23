setwd("C:/Programs and stuff/R Projects/Udemy Stuff/Machine-Learning-Dataset/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 14 - Logistic Regression")
# Logistic Regression

# Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]        #Predicting whether the audience will buy ticket just based on age and salary

# Encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)  #300 for training set and #100 for test set
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling  #Better to do feature scaling on classification cases
training_set[-3] = scale(training_set[-3])     #or training_set[,1:2] = scale(training_set[,1:2])
test_set[-3] = scale(test_set[-3])             #or test_set[,1:2] = scale(test_set[,1:2]) 

# Fitting Logistic Regression to the Training set, note our classifier is linear
classifier = glm(formula = Purchased ~ .,
                 family = binomial,
                 data = training_set)

# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test_set[-3]) #or test_set[,1:2]
y_pred = ifelse(prob_pred > 0.5, 1, 0)


# Making the Confusion Matrix  #To describe the performance of our classification model
cm = table(test_set[, 3], y_pred > 0.5)   #because the real results are in the 3rd column of the test set
cm

# Visualising the Training set results
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)  #Building grid for Age, -1 for min and +1 for max so the points won't get squeezed in the graph
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)  #Same here, for Salary column
grid_set = expand.grid(X1, X2)        #Last line to make matrix

colnames(grid_set) = c('Age', 'EstimatedSalary')

prob_set = predict(classifier, type = 'response', newdata = grid_set)  #Using classifier to predict position of each point/dot in graph
y_grid = ifelse(prob_set > 0.5, 1, 0)


plot(set[, -3],
     main = 'Logistic Regression (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))   #For the area shade
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))                #For each point     

# Visualising the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
setwd("C:/Programs and stuff/R Projects/Udemy Stuff/Machine-Learning-Dataset/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)")

# Artificial Neural Network

# Importing the dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[4:14]


#We don't need to encode target variable as factor (whether they stay in bank or not) because R recognises it categorical

# Encoding the categorical variables as factors
dataset$Geography = as.numeric(factor(dataset$Geography,            #Remember to convert to factor before turning it into numeric
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))


# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)   #8,000 obs to train, 2,000 to test
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Feature Scaling #Yes, we MUST apply feature scaling for ANN because it's computationally intensive, and required by package
training_set[-11] = scale(training_set[-11])
test_set[-11] = scale(test_set[-11])


# =============================   Fitting ANN to the Training set    ====================================
# install.packages('h2o')  #Helps connect us to connect to a computer system to run ANN
library(h2o)

#Connecting to a h2o server or instance. 
#nthreads = number of cores used, -1 is to use all the cores except 1
h2o.init(nthreads = -1)    


model = h2o.deeplearning(y = 'Exited',                           #note function used to deeplearning model
                         training_frame = as.h2o(training_set),  #Converting dataset into a h2o training frame
                         activation = 'Rectifier',               #Specifying activation function used. Best option mentioned by kirill
                         hidden = c(5,5),                        #Hidden layer sizes c(no. of hidden layer, no. of nodes in each hidden layer)
                         epochs = 100,                           #epochs = how many times the training set is passed through the ANN
                        
                         train_samples_per_iteration = -2)       #how many times we want to pass the training set through our ANN before weights adjusted.
                                                                 #"-2" means auto-tuning.


# Predicting the Test set results
y_pred = h2o.predict(model, newdata = as.h2o(test_set[-11]))     #predict function in the context of h2o
y_pred = (y_pred > 0.5)                                          #predict probability of customer leaving the bank, alternatively, y_pred = ifelse(y_pred > 0.5, 1, 0) 
y_pred = as.vector(y_pred)

# Making the Confusion Matrix
cm = table(test_set[, 11], y_pred)

# h2o.shutdown()

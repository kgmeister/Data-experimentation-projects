setwd("C:/Programs and stuff/R Projects/Udemy Stuff/Machine-Learning-Dataset/Machine Learning A-Z Template Folder/Part 4 - Clustering/Section 24 - K-Means Clustering")


# K-Means Clustering  #Note K Means++ for choosing the correct centroids

# Importing the dataset
dataset = read.csv('Mall_Customers.csv')
dataset = dataset[4:5]


# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Using the elbow method to find the optimal number of clusters

set.seed(6)        #Fixing a random seed number here so out K-Means won't take random seeds at every turn
wcss = vector()    #Initialising an empty factor


for (i in 1:10) wcss[i] = sum(kmeans(dataset, i)$withinss)    #Fitting kmeans to dataset with i clusters
plot(1:10,
     wcss,
     type = 'b',                                #"b" for Both points and lines
     main = paste('The Elbow Method'),
     xlab = 'Number of clusters',
     ylab = 'WCSS')


# Fitting K-Means to the dataset
set.seed(29)
kmeans = kmeans(x = dataset, centers = 5)
y_kmeans = kmeans$cluster

# Visualising the clusters
library(cluster)
clusplot(dataset,
         y_kmeans,
         lines = 0,
         shade = TRUE,                           #Ensure clusters are shaded according to their density
         color = TRUE,
         labels = 2,                             #So we have old points and clusters in the plot
         plotchar = FALSE,                       #We don't want different symbols for points in different clusters
         span = TRUE,                            #Plotting ellipses
         main = paste('Clusters of customers'),      
         xlab = 'Annual Income',
         ylab = 'Spending Score')

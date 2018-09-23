setwd("C:/Programs and stuff/R Projects/Udemy Stuff/Machine-Learning-Dataset/Machine Learning A-Z Template Folder/Part 5 - Association Rule Learning/Section 29 - Eclat")

# Eclat

# Data Preprocessing
# install.packages('arules')
library(arules)

dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Eclat on the dataset
rules = eclat(data = dataset, parameter = list(support = 0.003, minlen = 2))

# Visualising the results
inspect(sort(rules, by = 'support')[1:10])
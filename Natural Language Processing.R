setwd("C:/Programs and stuff/R Projects/Udemy Stuff/Machine-Learning-Dataset/Machine Learning A-Z Template Folder/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing")

# Natural Language Processing

# Importing the dataset
dataset_original = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)    

#Note the different read function, because it's a tsv file now. Note quote parameter as we don't want to pick ' ' as part of the text
#stringsAsFactors = FALSE because we're analysing the words themselves as contents
#and not a single entity, thus they cannot be factors.



# Cleaning the texts      #Remember bag of words concept
# install.packages('tm')
# install.packages('SnowballC')
library(tm)
library(SnowballC)        #Package for stopwords  


#We create a corpus, and clean the data inside it instead
#VCorpus function for creating a corpus for the text we want to clean (seen in bracket)
#We also clean it for one version of the word (i.e keeping the lowercase one and throwing out copies having uppercase)
corpus = VCorpus(VectorSource(dataset_original$Review))      


#**Cleaning step by step for all the reviews as the corpus is really large
#Content_transformer transform uppercases to lower cases, as shown by the tolower function here
corpus = tm_map(corpus, content_transformer(tolower))

#To access one element, double [[]] needed here
as.character(corpus[[1]])

#We remove numbers here because they're deemed irrelevant
corpus = tm_map(corpus, removeNumbers)


corpus = tm_map(corpus, removePunctuation)

#stopwords here contains all the non-relevant words like adjectives etc.
corpus = tm_map(corpus, removeWords, stopwords())


#To get the root word instead
corpus = tm_map(corpus, stemDocument)

#Remove extra space
corpus = tm_map(corpus, stripWhitespace)


# Creating the Bag of Words model          #Sparse matrix = matrix that has a lot of 0 in it
dtm = DocumentTermMatrix(corpus)           #dtm = Document Term Matrix
dtm = removeSparseTerms(dtm, 0.999)        #Removing sparse terms, keeping 99% of the most frequent words
dataset = as.data.frame(as.matrix(dtm))    #Converting dtm sparse matrix into a dataframe, as.matrix to be sure it gets imported as matrix
dataset$Liked = dataset_original$Liked     #Creating new "Liked" column to dataset


# Importing the dataset
#dataset = read.csv('Social_Network_Ads.csv')
#dataset = dataset[3:5]

# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#No feature scaling needed here because we're only dealing with 1's and 0's

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-692],          #Removing the independent variable 692
                          y = training_set$Liked,
                          ntree = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)                       #Predicting independent variable 692

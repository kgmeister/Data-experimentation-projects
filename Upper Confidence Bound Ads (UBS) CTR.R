setwd("C:/Programs and stuff/R Projects/Udemy Stuff/Machine-Learning-Dataset/Machine Learning A-Z Template Folder/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)")

# Upper Confidence Bound

# Importing the dataset
dataset = read.csv('Ads_CTR_Optimisation.csv')  #Which version of the ad to show?
View(dataset)

# Implementing UCB
N = 10000
d = 10
ads_selected = integer(0)                #Making the huge vector that contains the difference versions of the ad that was selected at each round
numbers_of_selections = integer(d)
sums_of_rewards = integer(d)                        #Sum of rewards at up to round n
total_reward = 0            
for (n in 1:N) {                                    #We are going through from round 1 to round N (10,000 here)
  ad = 0
  max_upper_bound = 0                               #Selecting the ad which has the highest upper bound
  for (i in 1:d) {                                  #At a specific round dealing with a specific version of the ad
    if (numbers_of_selections[i] > 0) {             #If ad is selected at least once
      average_reward = sums_of_rewards[i] / numbers_of_selections[i]   #Each taking the "i"th element of the vector
      delta_i = sqrt(3/2 * log(n) / numbers_of_selections[i])          
      upper_bound = average_reward + delta_i                           #Computing the upper confidence bound of the algorithm
    } else {
      upper_bound = 1e400
    }
    if (upper_bound > max_upper_bound) {                          
      max_upper_bound = upper_bound                     #We compute the upper bounds of each of the different ads at round N, equating the highest upper bound found to upper bound
      ad = i                                            #ecause i will have a specific value tied to a specific ad
    }
  }
  ads_selected = append(ads_selected, ad)
  numbers_of_selections[ad] = numbers_of_selections[ad] + 1
  reward = dataset[n, ad]
  sums_of_rewards[ad] = sums_of_rewards[ad] + reward
  total_reward = total_reward + reward
}

# Visualising the results
hist(ads_selected,
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of times each ad was selected')




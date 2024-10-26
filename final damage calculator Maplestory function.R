setwd('C:/Programs_and_stuff/R Projects/Random/exLotus_EDA')
exLotus <- read.csv('exLotus clear list.csv')
library(tidyverse)



# Vectorised form
fd_adv <- function(characterLevel, monsterLevel){
  compute_damage <- function(characterLevel, monsterLevel){
    levelDifference <- characterLevel - monsterLevel
    finalDamage <- 0
    
    if (levelDifference >=0 ){
    finalDamage <- 10 + (2 * levelDifference)
    finalDamage <- pmin(finalDamage, 20)
    }
    
    else if(levelDifference >= -5 & levelDifference <= -1){
    cappedPenalty <- ceiling(pmax((2.5*levelDifference), -12))
    bonus <- 2*(characterLevel - (monsterLevel-5))
    bonusPenalty <- (0.01 * bonus) * cappedPenalty
    finalDamage <- cappedPenalty + bonus + bonusPenalty
    }
    
    else {
    finalDamage <- ceiling(pmax(-100, 2.5 * levelDifference))
    }
    return(finalDamage)
    }
  # Vectorize the compute_damage function
    vectorized_compute_damage <- Vectorize(compute_damage)
    
    # Apply the function over vectors
    result <- vectorized_compute_damage(characterLevel, monsterLevel)
    
    return(result)
  }


fd_advantage <- fd_adv(exLotus$Level,285)

write.csv(exLotus, 'exLotus_ranking_final.csv')



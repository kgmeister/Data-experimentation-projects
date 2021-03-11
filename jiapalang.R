data <- c(1,2,3,4,5)
vector <- c()

test <- function(x){
  for(i in 1:length(x)){
    if(x[i]<=3){
      vector[i] <- x[i]*2
    }
    else vector[i] <- x[i]
  }
  return(vector)
}

test(data)

data <- c(1,2,3,4,5)

test <- function(x){
  container <- 0
  for(i in 1:length(x)){
  if(x[i]<=3){
        print (2*x[i])
}
else (print (x[i]))
  }
}

test(data)

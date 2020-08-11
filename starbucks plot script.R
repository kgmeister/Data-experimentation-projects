getwd()
setwd("C:/Programs and stuff/R Projects/Random")
disloyal<- read.csv("disloyalcustomers.csv")
loyal <- read.csv("loyalcustomers.csv")

customer<- cbind(loyal,disloyal[,"Freq"])
customer<-customer[,2:4]

colnames(customer) <- cbind("Distance", "Loyal_customers", "Disloyal_Customers")  


ggplot(customer, aes(x=Distance)) +
         geom_smooth(aes(y=Loyal_customers)) +
  geom_smooth(aes(y=Disloyal_Customers), color="red", linetype= "twodash") +
  ggtitle("Number of loyal and disloyal customers vs distance travelled to Starbucks") +
  xlab("Distance") + ylab("Number of customers") 
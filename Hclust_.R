###Performing clustering for the airlines data to obtain optimum number of clusters. 

east <- read.csv("/Users/jobinsamuel/Desktop/EastWest.csv") #Read data
sum(is.na(east))   #Checking for na values
summary(east)      #To know the info of  min , max etc

east_data <- scale(east[ , ])  #Standardising the data
summary(east_data)

est <- dist(east_data, method = "euclidean") #Finding the distance

estw <- hclust(est, method = "complete")
#Dendrogram
plot(estw)
plot(estw,hang = -1)
easwst <- cutree(estw, k = 2) # Cut tree into 4 clusters

#Borders for the clusters
rect.hclust(estw, k = 2, border = "blue")

#Converting the the number of clusters which was cut to matrix
mem <- as.matrix(easwst)
#Converting the matrix which was created to data frame
feaswst <- data.frame(mem,east)

aggregate(east, by = list(mem), FUN = mean)

###Performing clustering for the crime data and identify the number of clusters formed 

crime <- read.csv("/Users/jobinsamuel/Desktop/Assignments/Hierarchial Clustering/Dataset_Assignment Clustering/crime_data.csv") #Read data
sum(is.na(crime))   #Checking for na values
summary(crime)      #To know the info of  min , max etc

crime_data <- scale(crime[ ,2:5])  #Standardising the data
summary(crime_data)

crm <- dist(crime_data, method = "euclidean") #Finding the distance

crmw <- hclust(crm, method = "complete")
#Dendrogram
plot(crmw)
plot(crmw,hang = -1)
crim <- cutree(crmw, k = 4) # Cut tree into 4 clusters

#Borders for the clusters
rect.hclust(crmw, k = 4, border = "purple")

#Converting the the number of clusters which was cut to matrix
cir <- as.matrix(crim)
#Converting the matrix which was created to data frame
fcrime <- data.frame(cir,crime)
# Aggregate by mean of each cluster
aggregate(crime, by = list(cir), FUN = mean) #Ignore warnings

###Perform clustering analysis on the telecom data set. The data is a mixture of both categorical and numerical data.
### It consists the number of customers who churn. Derive insights and get possible information on factors that may affect the churn decision.

library(readxl)
tele <- read_excel("/Users/jobinsamuel/Downloads/Dataset_Assignment Clustering/Telco_customer_churn.xlsx")
sum(is.na(tele))
summary(tele)

telc <- tele[,c(-2:-5,-18:-20,-23)]#Removing unecessary columns
#Creating Dummies for character data 
library(fastDummies)
dum <- dummy_cols(telc, select_columns = c("Offer","Phone Service","Multiple Lines","Internet Service","Internet Type","Online Security","Online Backup","Device Protection Plan","Premium Tech Support","Unlimited Data","Contract","Payment Method"),
                  remove_first_dummy = TRUE,remove_most_frequent_dummy = FALSE,remove_selected_columns = TRUE)


nm_dum <-scale(dum[,2:30]) #Normalizing the data 
library(cluster)
gower_dist <- daisy(nm_dum[, -1],
                    metric = "euclidean",
                    type = list(logratio = 3))
gower_dist

#Dendrogram
plot(gower_dist)
plot(gower_dist,hang = -1)
crim <- cutree(crmw, k = 4) # Cut tree into 4 clusters

#Borders for the clusters
rect.hclust(crmw, k = 4, border = "purple")

#Converting the the number of clusters which was cut to matrix
cir <- as.matrix(crim)
#Converting the matrix which was created to data frame
fcrime <- data.frame(cir,crime)
# Aggregate by mean of each cluster
aggregate(crime, by = list(cir), FUN = mean) #Ignore warnings

###Performing clustering on mixed data convert the categorical variables to numeric by using dummies or Label Encoding and perform normalization techniques.
### The data set consists details of customers related to auto insurance.
auto <-read.csv ("/Users/jobinsamuel/Desktop/Assignments/Hierarchial Clustering/Dataset_Assignment Clustering/AutoInsurance.csv")
auto <- aut[,c(-1:-2,-7:-8)]#Removing unecessary columns
#Creating Dummies for character data 
library(fastDummies)
dum_a <-  dummy_cols(auto, select_columns = c("Response","Coverage","Education","Gender","Location.Code","Marital.Status","Policy.Type","Policy","Renew.Offer.Type","Sales.Channel","Vehicle.Class","Vehicle.Size"),
                     remove_first_dummy = TRUE,remove_most_frequent_dummy = FALSE,remove_selected_columns = TRUE)

nmat <- scale(dum_a[ , ]) #Normalizing the data 
summary(nmat)

at <- dist(nmat, method = "euclidean") #Finding the distance

atw <- hclust(at, method = "complete")
#Dendrogram
plot(atw)
plot(atw,hang = -1)
autoins <- cutree(atw, k = 3) # Cut tree into 4 clusters
#Borders for the clusters
rect.hclust(atw, k = 4, border = "red")

#Converting the the number of clusters which was cut to matrix
ati <- as.matrix(autoins)
#Converting the matrix which was created to data frame
fatins <- data.frame(ati,auto)
# Aggregate by mean of each cluster
aggregate(auto, by = list(ati), FUN = mean) #Ignore warnings

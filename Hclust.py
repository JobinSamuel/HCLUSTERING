import pandas as pd       #import pandas
import matplotlib.pylab as plt  #Import matplotlib.pylab for plots

##Question -1 
air = pd.read_csv("/Users/jobinsamuel/Desktop/EastWest.csv")  #Using read_csv  to read .csv file

air.describe() #Used to get info about min max mean etc

# Normalization function 
def norm(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

#Normalizing data
air_norm = norm(air.iloc[:, 1:])
air_norm.describe()    #To check if normilization was proper

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage #TO GET A DENDROGRAM WE USE THESE FUNCTIONS
import scipy.cluster.hierarchy as sch 

z = linkage(air_norm, method = "complete")  

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Balance');plt.ylabel('Bonus_miles')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()  #Plot is show since it consists of 4000 data points it takes time to get the output of dendrogram

# Now applying AgglomerativeClustering 
from sklearn.cluster import AgglomerativeClustering

#choosing 4 clusters from the above dendrogram

complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(air_norm) 
complete.labels_

air_labels = pd.Series(complete.labels_) #Using labels

air['cluster'] = air_labels # creating a new column and assigning it to air dataset

air1 = air.iloc[:, [12,0,1,2,3,4,5,6,7,8,9,10,11]] #Rearragnging the columns so that cluster comes first
air1.head() #Gives the first five data

# Aggregate by mean of each cluster
air1.iloc[:, 2:].groupby(air1.cluster).mean() #Using groupby function and finding the mean  to column 2 to 13  


####Question-2
vdm = pd.read_csv("/Users/jobinsamuel/Desktop/Assignments/Hierarchial Clustering/Dataset_Assignment Clustering/crime_data.csv") #Read the data
vdm.describe()       #Used to get info about min max mean etc
vdm.isna().sum()    #Checking for any NA values if present

# Normalization function 
def nm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalizing data
vd_norm = nm_func(vdm.iloc[:, 1:])
vd_norm.describe()  #To check if normilization was proper

from scipy.cluster.hierarchy import linkage #TO GET A DENDROGRAM WE USE THESE FUNCTIONS
import scipy.cluster.hierarchy as sch 

w = linkage(vd_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('H Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(w, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()  #Shows the dendrogram


# Now applying AgglomerativeClustering 
from sklearn.cluster import AgglomerativeClustering

#choosing 4 clusters from the above dendrogram
vdm_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(vd_norm) 
vdm_complete.labels_

vdm_labels = pd.Series(vdm_complete.labels_) #Using labels

vdm['clust'] = vdm_labels # creating a new column and assigning it to vdm dataset

vdm.head() #Gives the first five data

# Aggregate mean of each cluster
vdm.iloc[:, :5].groupby(vdm.clust).mean()#Using groupby function and finding the mean  to column 0 to 5 


####Question-3

chrn=pd.read_excel("/Users/jobinsamuel/Downloads/Dataset_Assignment Clustering/Telco_customer_churn.xlsx") #Read the data

chrn.describe()  #Used to get info about min max mean etc
chrn.isna().sum()  #Checking for NA values if present

chrn.drop(columns=['Customer ID','Count','Quarter','Referred a Friend','Number of Referrals','Streaming Movies','Streaming Music','Paperless Billing'],inplace=True)  #Droping unnecesary columns Inplace = true used to refelct the dropped na values to the dataframe

#Creates dummies
df_n =  pd.get_dummies(chrn,columns=['Phone Service','Multiple Lines','Offer','Internet Service','Online Security','Streaming TV','Online Backup','Device Protection Plan','Premium Tech Support','Contract','Payment Method','Unlimited Data','Internet Type']) #Creates dummies

df_n.describe()

#Normalization Function
def nm(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalizing data 
chrn_norm = nm(df_n.iloc[:, :])
chrn_norm.describe()  #To check if normilization was proper

from scipy.cluster.hierarchy import linkage #TO GET A DENDROGRAM WE USE THESE FUNCTIONS
import scipy.cluster.hierarchy as sch 

ch = linkage(df_n, method = "complete")
# Dendrogram
plt.figure(figsize=(15, 8));plt.title('H Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(ch, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()   #Shows the dendrogram 
#Here while displaying dendrogram it takes 2-8minutes to load

#Gower's dissimilarity
import gower   #Using gower to find the distance between the clusters
import numpy as np
X = np.asarray(df_n) #Converting into array format
X
type(X)

y = gower.gower_matrix(X)
y

# Now applying AgglomerativeClustering 
from sklearn.cluster import AgglomerativeClustering

#choosing 3 clusters from the above dendrogram
hi_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete').fit(chrn_norm) 
hi_complete.labels_

cl_labels = pd.Series(hi_complete.labels_)

chrn['clust'] = cl_labels  # creating a new column and assigning it to new column 


chrn.head() #Gives the first five data


# Aggregate mean of each cluster
chrn.iloc[:, 1:22].groupby(chrn.clust).mean() #Using groupby function and finding the mean  to column 1 to 22 


###Question-4

auto = pd.read_csv("/Users/jobinsamuel/Desktop/Assignments/Hierarchial Clustering/Dataset_Assignment Clustering/AutoInsurance.csv") #Read data
auto.describe()      #Used to get info about min max mean etc
auto.isna().sum()

auto.drop(columns=['Customer','Effective To Date'],inplace=True)#Droping unnecesary columns Inplace = true used to refelct the dropped na values to the dataframe

#Creating dummies
au = pd.get_dummies(auto,columns=['State','Response','Coverage','Education','EmploymentStatus','Gender','Location Code','Marital Status','Policy Type','Policy','Renew Offer Type','Sales Channel','Vehicle Class','Vehicle Size'])

au.describe()

def at(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
auto_norm = at(au.iloc[:, 1:])
auto_norm.describe()  #To check if normilization was proper

from scipy.cluster.hierarchy import linkage #TO GET A DENDROGRAM WE USE THESE FUNCTIONS
import scipy.cluster.hierarchy as sch 

at = linkage(au, method = "single")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('H Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(at, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()   #Shows the dendrogram 
#Here while displaying dendrogram it takes 2-5minutes to load

# Now applying AgglomerativeClustering 

from sklearn.cluster import AgglomerativeClustering

#choosing 3 clusters from the above dendrogram

at_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete').fit(auto_norm) 
at_complete.labels_

at_labels = pd.Series(at_complete.labels_)

auto['cluster'] = at_labels  # creating a new column and assigning it to new column 


auto.head() #Gives the first five data


# Aggregate mean of each cluster
auto.iloc[:, 1:22].groupby(auto.cluster).mean()#Using groupby function and finding the mean  to column 1 to 22 


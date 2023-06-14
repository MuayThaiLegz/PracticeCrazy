# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 06:20:43 2023

@author: abdullahalbinsaleh

This version uses normal random data splitting technique 
"""

#-------------------------import librarires-------------------------#
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier 
from sklearn.neighbors import KNeighborsClassifier
#from xgboost import XGBClassifier
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
#-------------------------Load dataset-------------------------#
#moneyFlow = pd.read_excel("/Users/Al-Ikhwan/Library/CloudStorage/Dropbox/Data Capstone/Dataset Complete/moneyFlow.xlsx")

moneyFlow = pd.read_excel("/Users/JBarr/OneDrive/Documents/FinTech/TutorSection/NewT/ogml/wa/moneyFlow.xlsx")
moneyFlow = moneyFlow.drop('Unnamed: 0', axis=1)

#-------------------------Data Exploration-------------------------#
moneyFlow.head()
moneyFlow.info()
moneyFlow.columns


#-------------------------Data Preparation & Pre-Processing-------------------------#
#Drop missing values 
moneyFlow.dropna(inplace=True)

#drop $ sign from Spot feature 
# convert the 'Spot' column to a string
moneyFlow['Spot'] = moneyFlow['Spot'].astype(str)
moneyFlow['Spot'] = moneyFlow['Spot'].str.replace("$", "")
# convert the 'Spot' column back to float
moneyFlow['Spot'] = moneyFlow['Spot'].astype(float)

#convert Date and Expiration to timestamp 
moneyFlow["Date"] = pd.to_datetime(moneyFlow["Date"])
moneyFlow["Expiration"] = pd.to_datetime(moneyFlow["Expiration"])


#-------------------------Feature Engineering -------------------------#
#Split Details feature into two features 
moneyFlow[["Size", "Price"]] = moneyFlow["Details"].str.split("@", expand=True)
#drop the Details feature 
moneyFlow.drop(columns=["Details"], inplace=True)

#Find options duration in days 
moneyFlow["Duration"] = (moneyFlow["Expiration"]-moneyFlow["Date"]).dt.days
#drop the Date and Expiration deatuers 
moneyFlow.drop(columns=["Date"], inplace=True)
moneyFlow.drop(columns=["Expiration"], inplace=True)


#Convert categoircal variables into dummy variables 
moneyFlow = pd.get_dummies(moneyFlow, columns=["Type", "Execution", "C/P"])


#-------------------------Feature Selection Using Domain Knowledge -------------------------#
#Drop useless predictors using domain knowledge
moneyFlow.drop(columns=["Time", "Ticker", "Price", "Size", "Volume", "Open Interest"], inplace=True)

#new data set columns 
moneyFlow.info()
moneyFlow.columns

#-------------------------Cluestring-------------------------#


#clustering based  on Premium and Type of Trade
premType = moneyFlow.iloc[:,[2,5,6]]
print(premType)
#scale the data
scaler = StandardScaler()
premType_scaled = scaler.fit_transform(premType)



#2 clusters 
premType_2_Clusters = KMeans(n_clusters=2,random_state=0)
km = premType_2_Clusters.fit_predict(premType_scaled)

# Plot each of the clusters

plt.scatter(premType_scaled[km == 0, 0],
            premType_scaled[km == 0, 1],
            s=50,
            c='lightgreen', edgecolor='black',marker='s', label='Cluster 1')

plt.scatter(premType_scaled[km == 1, 0],
            premType_scaled[km == 1, 1],
            s=50,
            c='orange', edgecolor='black',marker='o', label='Cluster 2')

plt.scatter(premType_2_Clusters.cluster_centers_[:, 0],
            premType_2_Clusters.cluster_centers_[:, 1],
            s=250, marker='*', c='red', edgecolor='black', label='Centriods'
            )

plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()


#4 clusters 


premType_4_Clusters = KMeans(n_clusters=4, random_state=0)
km = premType_4_Clusters.fit_predict(premType_scaled)

# Plot each of the clusters
plt.scatter(premType_scaled[km == 0, 0],
            premType_scaled[km == 0, 1],
            s=50,
            c='lightgreen',
            edgecolor='black',
            marker='s',
            label='Cluster 1')

plt.scatter(premType_scaled[km == 1, 0],
            premType_scaled[km == 1, 1],
            s=50,
            c='orange',
            edgecolor='black',
            marker='o',
            label='Cluster 2')

plt.scatter(premType_scaled[km == 2, 0],
            premType_scaled[km == 2, 1],
            s=50,
            c='blue',
            edgecolor='black',
            marker='v',
            label='Cluster 3')

plt.scatter(premType_scaled[km == 3, 0],
            premType_scaled[km == 3, 1],
            s=50,
            c='purple',
            edgecolor='black',
            marker='*',
            label='Cluster 4')

for i, centroid in enumerate(premType_4_Clusters.cluster_centers_):
    plt.scatter(centroid[0], centroid[1],
                s=250,
                marker='*',
                c='red',
                edgecolor='black',
                label=f'Centroid {i+1}')

plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()

#calculate the Silhouette score for each cluster 
#this score ranges from âˆ’1 to +1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.

premType_2_Clusters_silhouette_avg = silhouette_score(premType_scaled, premType_2_Clusters.labels_)
print('Silhouette score:', premType_2_Clusters_silhouette_avg )
#score 0.930

premType_4_Clusters_silhouette_avg = silhouette_score(premType_scaled, premType_4_Clusters.labels_)
print('Silhouette score:', premType_4_Clusters_silhouette_avg )
#score 0.944


#cluster membership
membership = pd.Series(premType_4_Clusters.labels_, index = moneyFlow.index)
moneyFlow['membership'] = membership
moneyFlow.head()
moneyFlow["membership"].unique()

unique_clusters = moneyFlow["membership"].unique()


#-------------------------Dataset Based on Clustering -------------------------#

#create four data sets according to membership and display info for each one

moneyFlow_cluster_1 = moneyFlow[moneyFlow["membership"] == 0]
moneyFlow_cluster_2 = moneyFlow[moneyFlow["membership"] == 1]
moneyFlow_cluster_3 = moneyFlow[moneyFlow["membership"] == 2]
moneyFlow_cluster_4 = moneyFlow[moneyFlow["membership"] == 3]
for df in [moneyFlow_cluster_1,moneyFlow_cluster_2,moneyFlow_cluster_3,moneyFlow_cluster_4]:
    print(df.info())

# This will create a boxplot of the 'Premium' variable for each cluster
fig, ax = plt.subplots(figsize=(12, 8))

data_to_plot = [moneyFlow_cluster_1['Prem'], moneyFlow_cluster_2['Prem'], 
                moneyFlow_cluster_3['Prem'], moneyFlow_cluster_4['Prem']]

labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']

bplot = ax.boxplot(data_to_plot,
                     vert=True,  
                     notch=True,
                     patch_artist=True,  
                     labels=labels)  
ax.set_title('Premium Distributions per Cluster')
ax.set_xlabel('Cluster')
ax.set_ylabel('Premium')

# fill with colors
colors = ['skyblue', 'lightgreen', 'pink', 'lightyellow']

for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)



ax.yaxis.grid(True)
ax.xaxis.grid(True)


plt.show()

#-------------------------Data Split-------------------------#
# Split the data into X and y for cluster 1
X_cluster_1 = moneyFlow_cluster_1.drop('Sentiment', axis=1)
y_cluster_1= moneyFlow_cluster_1['Sentiment']
# Split the data into training and testing sets for cluster 1
X_train_cluster_1, X_test_cluster_1, y_train_cluster_1, y_test_cluster_1 = train_test_split(X_cluster_1, y_cluster_1, test_size=0.3, random_state=42)

# Split the data into X and y for cluster 2
X_cluster_2 = moneyFlow_cluster_2.drop('Sentiment', axis=1)
y_cluster_2= moneyFlow_cluster_2['Sentiment']
# Split the data into training and testing sets for cluster 1
X_train_cluster_2, X_test_cluster_2, y_train_cluster_2, y_test_cluster_2 = train_test_split(X_cluster_2, y_cluster_2, test_size=0.3, random_state=42)

# Split the data into X and y for cluster 3
X_cluster_3 = moneyFlow_cluster_3.drop('Sentiment', axis=1)
y_cluster_3= moneyFlow_cluster_3['Sentiment']
# Split the data into training and testing sets for cluster 1
X_train_cluster_3, X_test_cluster_3, y_train_cluster_3, y_test_cluster_3 = train_test_split(X_cluster_3, y_cluster_3, test_size=0.3, random_state=42)

# Split the data into X and y for cluster 4
X_cluster_4 = moneyFlow_cluster_4.drop('Sentiment', axis=1)
y_cluster_4= moneyFlow_cluster_4['Sentiment']
# Split the data into training and testing sets for cluster 1
X_train_cluster_4, X_test_cluster_4, y_train_cluster_4, y_test_cluster_4 = train_test_split(X_cluster_4, y_cluster_4, test_size=0.3, random_state=42)


# ---------------------Helper Functions----------------------------#


# Function for feature scaling
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)   # Use transform, not fit_transform on test set
    return X_train_scaled, X_test_scaled

# Function for model fitting and prediction
def fit_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

# Function for model evaluation
def evaluate_model(y_test, y_pred, model_name):
    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"{model_name} Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


        
#-------------------------Feature Scaling-------------------------#
X_train_cluster_1, X_test_cluster_1 = scale_features(X_train_cluster_1, X_test_cluster_1)

X_train_cluster_2, X_test_cluster_2 = scale_features(X_train_cluster_2, X_test_cluster_2)

X_train_cluster_3, X_test_cluster_3 = scale_features(X_train_cluster_3, X_test_cluster_3)

X_train_cluster_4, X_test_cluster_4 = scale_features(X_train_cluster_4, X_test_cluster_4)


#---------- Looking at K=4 ----------------#

#-------------------------Logisitc Regression-------------------------#
    
#-------------------------Logisitc Regression Cluster 1-------------------------#
lr = LogisticRegression(max_iter=10000)
lr_pred_cluster_1 = fit_predict(lr, X_train_cluster_1, y_train_cluster_1, X_test_cluster_1)
evaluate_model(y_test_cluster_1, lr_pred_cluster_1, "Logistic Regression")

#-------------------------Logisitc Regression Cluster 2-------------------------#
lr = LogisticRegression(max_iter=10000)
lr_pred_cluster_2 = fit_predict(lr, X_train_cluster_2, y_train_cluster_2, X_test_cluster_2)
evaluate_model(y_test_cluster_2, lr_pred_cluster_2, "Logistic Regression")

#-------------------------Logisitc Regression Cluster 3-------------------------#
lr = LogisticRegression(max_iter=10000)
lr_pred_cluster_3 = fit_predict(lr, X_train_cluster_3, y_train_cluster_3, X_test_cluster_3)
evaluate_model(y_test_cluster_3, lr_pred_cluster_3, "Logistic Regression")

#-------------------------Logisitc Regression Cluster 4-------------------------#
lr = LogisticRegression(max_iter=10000)
lr_pred_cluster_4 = fit_predict(lr, X_train_cluster_4, y_train_cluster_4, X_test_cluster_4)
evaluate_model(y_test_cluster_4, lr_pred_cluster_4, "Logistic Regression")


# ----------------------Random Forest Classifier----------------------

# ----------------------Random Forest Classifier for Cluster 1----------------------
rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
rf_pred_cluster_1 = fit_predict(rf, X_train_cluster_1, y_train_cluster_1, X_test_cluster_1)
evaluate_model(y_test_cluster_1, rf_pred_cluster_1, "Random Forest")

# ----------------------Random Forest Classifier for Cluster 2----------------------
rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
rf_pred_cluster_2 = fit_predict(rf, X_train_cluster_2, y_train_cluster_2, X_test_cluster_2)
evaluate_model(y_test_cluster_2, rf_pred_cluster_2, "Random Forest")

# ----------------------Random Forest Classifier for Cluster 3----------------------
rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
rf_pred_cluster_3 = fit_predict(rf, X_train_cluster_3, y_train_cluster_3, X_test_cluster_3)
evaluate_model(y_test_cluster_3, rf_pred_cluster_3, "Random Forest")

# ----------------------Random Forest Classifier for Cluster 4----------------------
rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
rf_pred_cluster_4 = fit_predict(rf, X_train_cluster_4, y_train_cluster_4, X_test_cluster_4)
evaluate_model(y_test_cluster_4, rf_pred_cluster_4, "Random Forest")


#-------------------------Neural Netowrk-------------------------#

# ----------------------Neural Network for Cluster 1----------------------
nn = MLPClassifier(hidden_layer_sizes=(64,),max_iter=1000, random_state=42)
nn_pred_cluster_1 = fit_predict(nn, X_train_cluster_1, y_train_cluster_1, X_test_cluster_1)
evaluate_model(y_test_cluster_1, nn_pred_cluster_1, "Neural Network")

# ----------------------Neural Network for Cluster 2----------------------
nn = MLPClassifier(hidden_layer_sizes=(64,),max_iter=1000, random_state=42)
nn_pred_cluster_2 = fit_predict(nn, X_train_cluster_2, y_train_cluster_2, X_test_cluster_2)
evaluate_model(y_test_cluster_2, nn_pred_cluster_2, "Neural Network")

# ----------------------Neural Network for Cluster 3----------------------
nn = MLPClassifier(hidden_layer_sizes=(64,),max_iter=1000, random_state=42)
nn_pred_cluster_3 = fit_predict(nn, X_train_cluster_3, y_train_cluster_3, X_test_cluster_3)
evaluate_model(y_test_cluster_3, nn_pred_cluster_3, "Neural Network")

# ----------------------Neural Network for Cluster 4----------------------
nn = MLPClassifier(hidden_layer_sizes=(64,),max_iter=1000, random_state=42)
nn_pred_cluster_4 = fit_predict(nn, X_train_cluster_4, y_train_cluster_4, X_test_cluster_4)
evaluate_model(y_test_cluster_4, nn_pred_cluster_4, "Neural Network")


#-------------------------KNN-------------------------#

# ----------------------KNN for Cluster 1----------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn_pred_cluster_1 = fit_predict(knn, X_train_cluster_1, y_train_cluster_1, X_test_cluster_1)
evaluate_model(y_test_cluster_1, knn_pred_cluster_1, "K-Nearest Neighbors")

# ----------------------KNN for Cluster 2----------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn_pred_cluster_2 = fit_predict(knn, X_train_cluster_2, y_train_cluster_2, X_test_cluster_2)
evaluate_model(y_test_cluster_2, knn_pred_cluster_2, "K-Nearest Neighbors")

# ----------------------KNN for Cluster 3----------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn_pred_cluster_3 = fit_predict(knn, X_train_cluster_3, y_train_cluster_3, X_test_cluster_3)
evaluate_model(y_test_cluster_3, knn_pred_cluster_3, "K-Nearest Neighbors")

# ----------------------KNN for Cluster 4----------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn_pred_cluster_4 = fit_predict(knn, X_train_cluster_4, y_train_cluster_4, X_test_cluster_4)
evaluate_model(y_test_cluster_4, knn_pred_cluster_4, "K-Nearest Neighbors")


#---------- Now looking at K=2 ----------------#


#cluster membership
membershipk2 = pd.Series(premType_2_Clusters.labels_, index = moneyFlow.index)
moneyFlow['membershipk2'] = membershipk2
moneyFlow.head()
unique_clustersk2 = moneyFlow["membershipk2"].unique()


#-------------------------Dataset Based on Clustering -------------------------#
#create four data sets according to membership 
moneyFlow_clusterk2_1 = moneyFlow[moneyFlow["membershipk2"] == 0]
moneyFlow_clusterk2_2 = moneyFlow[moneyFlow["membershipk2"] == 1]

for df in [moneyFlow_clusterk2_1, moneyFlow_clusterk2_2]:
    print(df.info())
    
fig, ax = plt.subplots(figsize=(12, 8))

data_to_plot = [moneyFlow_clusterk2_1['Prem'], moneyFlow_clusterk2_2['Prem']]

labels = ['Cluster 1', 'Cluster 2']


bplot = ax.boxplot(data_to_plot,
                     vert=True,  
                     patch_artist=True,  
                     notch=True,
                     labels=labels)  
ax.set_title('Premium Distributions per Cluster')
ax.set_xlabel('Cluster')
ax.set_ylabel('Premium')

colors = ['skyblue', 'lightgreen']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

ax.yaxis.grid(True)
ax.xaxis.grid(True)

plt.show()


#-------------------------Data Split-------------------------#
# Split the data into X and y for cluster 1
X_clusterk2_1 = moneyFlow_clusterk2_1.drop('Sentiment', axis=1)
y_clusterk2_1= moneyFlow_clusterk2_1['Sentiment']
# Split the data into training and testing sets for cluster 1
X_train_clusterk2_1, X_test_clusterk2_1, y_train_clusterk2_1, y_test_clusterk2_1 = train_test_split(X_clusterk2_1, y_clusterk2_1, test_size=0.3, random_state=42)

# Split the data into X and y for cluster 2
X_clusterk2_2 = moneyFlow_clusterk2_2.drop('Sentiment', axis=1)
y_clusterk2_2 = moneyFlow_clusterk2_2['Sentiment']
# Split the data into training and testing sets for cluster 1
X_train_clusterk2_2, X_test_clusterk2_2, y_train_clusterk2_2, y_test_clusterk2_2 = train_test_split(X_clusterk2_2, y_clusterk2_2, test_size=0.3, random_state=42)


#-------------------------Feature Scaling-------------------------#
X_train_clusterk2_1, X_test_clusterk2_1 = scale_features(X_train_clusterk2_1, X_test_clusterk2_1)

X_train_clusterk2_2, X_test_clusterk2_2 = scale_features(X_train_clusterk2_2, X_test_clusterk2_2)

#---------- Looking at K=2 ----------------#

#-------------------------Logisitc Regression-------------------------#

#-------------------------Logisitc Regression Cluster 1-------------------------#
lr = LogisticRegression(max_iter=10000)
lr_pred_clusterk2_1 = fit_predict(lr, X_train_clusterk2_1, y_train_clusterk2_1, X_test_clusterk2_1)
evaluate_model(y_test_clusterk2_1, lr_pred_clusterk2_1, "Logistic Regression")

#-------------------------Logisitc Regression Cluster 2-------------------------#
lr = LogisticRegression(max_iter=10000)
lr_pred_clusterk2_2 = fit_predict(lr, X_train_clusterk2_2, y_train_clusterk2_2, X_test_clusterk2_2)
evaluate_model(y_test_clusterk2_2, lr_pred_clusterk2_2, "Logistic Regression")

# ----------------------Random Forest Classifier----------------------

#-------------------------Random Forest Classifier for Cluster 1-------------------------
rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
rf_pred_clusterk2_1 = fit_predict(rf, X_train_clusterk2_1, y_train_clusterk2_1, X_test_clusterk2_1)
evaluate_model(y_test_clusterk2_1, rf_pred_clusterk2_1, "Random Forest")

#-------------------------Random Forest Classifier for Cluster 2-------------------------
rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
rf_pred_clusterk2_2 = fit_predict(rf, X_train_clusterk2_2, y_train_clusterk2_2, X_test_clusterk2_2)
evaluate_model(y_test_clusterk2_2, rf_pred_clusterk2_2, "Random Forest")


#-------------------------Neural Netowrk-------------------------#

#-------------------------Neural Network Classifier for Cluster 1-------------------------
nn = MLPClassifier(hidden_layer_sizes=(64,),max_iter=1000, random_state=42)
nn_pred_clusterk2_1 = fit_predict(nn, X_train_clusterk2_1, y_train_clusterk2_1, X_test_clusterk2_1)
evaluate_model(y_test_clusterk2_1, nn_pred_clusterk2_1, "Neural Network")

#-------------------------Neural Network Classifier for Cluster 2-------------------------
nn = MLPClassifier(hidden_layer_sizes=(64,),max_iter=1000, random_state=42)
nn_pred_clusterk2_2 = fit_predict(nn, X_train_clusterk2_2, y_train_clusterk2_2, X_test_clusterk2_2)
evaluate_model(y_test_clusterk2_2, nn_pred_clusterk2_2, "Neural Network")


#-------------------------KNN-------------------------#

#-------------------------K Nearest Neighbors Classifier for Cluster 1-------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn_pred_clusterk2_1 = fit_predict(knn, X_train_clusterk2_1, y_train_clusterk2_1, X_test_clusterk2_1)
evaluate_model(y_test_clusterk2_1, knn_pred_clusterk2_1, "K Nearest Neighbors")

#-------------------------K Nearest Neighbors Classifier for Cluster 2-------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn_pred_clusterk2_2 = fit_predict(knn, X_train_clusterk2_2, y_train_clusterk2_2, X_test_clusterk2_2)
evaluate_model(y_test_clusterk2_2, knn_pred_clusterk2_2, "K Nearest Neighbors")

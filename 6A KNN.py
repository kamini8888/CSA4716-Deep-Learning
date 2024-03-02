import numpy as np
import pandas as pd

dataset = pd.read_csv("C:/Users/kamini/Downloads/machine learning/IRIS.csv")
"""
The breast cancer dataset has the following features: Sample code number, Clump Thickness, Uniformity of Cell Size, 
Uniformity of Cell Shape, Marginal Adhesion, Single Epithelial Cell Size, Bare Nuclei, Bland Chromatin,
 Normal Nucleoli, Mitosis, Class.
"""
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
dataset.shape
#splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
#Feature Scaling
"""
Feature scaling is the process of converting the data into a given range. 
In this case, the standard scalar technique is used.
"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#Training the K-Nearest Neighbors (K-NN) Classification model on the Training set
"""
Once the dataset is scaled, next, the K-Nearest Neighbors (K-NN) classifier algorithm is used to create a model. 
The hyperparameters such as n_neighbors, metric, and p are set to 5, Minkowski, and 2 respectively.
 The remaining hyperparameters are set to default values.
"""
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

classifier.fit(X_train, y_train)
"""
Display the results (confusion matrix and accuracy)
Here evaluation metrics such as confusion matrix and accuracy are used to evaluate the performance of the model built using a decision tree classifier.
"""
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

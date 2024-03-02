import numpy as np
import pandas as pd
#Importing the dataset
"""
Next, we import or read the dataset. Click here to download the breast cancer dataset used in this implementation.
 After reading the dataset, divide the dataset into concepts and targets. Store the concepts into X and 
 targets into y.
"""
dataset = pd.read_csv("C:/Users/kamini/Downloads/machine learning/IRIS.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
"""
Splitting the dataset into the Training set and Test set
Once the dataset is read into the memory, next, divide the dataset into two parts, training and 
testing using the train_test_split function from sklearn.
 The test_size and random_state attributes are set to 0.25 and 0 respectively. 
 You can change these attributes as per your requirements.
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Feature Scaling
"""
Feature scaling is the process of converting the data into a min-max range. In this case,
 the standard scalar method is used.
"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""
Training the Naive Bayes Classification model on the Training set
Once the dataset is scaled, next, the Naive Bayes classifier algorithm is used to create a model. 
The GaussianNB function is imported from sklearn.naive_bayes library. The hyperparameters such as kernel, 
and random_state to linear, and 0 respectively. The remaining hyperparameters of the support vector machine
 algorithm are set to default values.
"""
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Naive Bayes classifier model
GaussianNB(priors=None, var_smoothing=1e-09)

#Display the results (confusion matrix and accuracy)
"""
Here evaluation metrics such as confusion matrix and accuracy are used to evaluate the performance of 
the model built using a decision tree classifier.
"""
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

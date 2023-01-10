from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np


data=load_breast_cancer()
X=data.data
y=data.target
feature_name=data.feature_names
response_vector=data.target_names
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

classifier_knn = KNeighborsClassifier(n_neighbors = 3)
classifier_knn.fit(X_train, y_train)
y_pred = classifier_knn.predict(X_test)
sample =X[0:25]
sample.reshape(1, -1)
print(sample)
preds = classifier_knn.predict(sample)
for p in preds:
    pred_species=response_vector[p]
print(pred_species)
print(feature_name)
print(response_vector)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
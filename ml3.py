from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt 

data=load_wine()

X=data.data
y=data.target
print(X)
print(y)
response_vector=data.target_names
print(data.feature_names)
print(response_vector)
feature_name=data.feature_names
response_vector=data.target_names
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.8,random_state=1)
sample =X[0:1]
classifier_knn = KNeighborsClassifier(n_neighbors = 3)
classifier_knn.fit(X_train, y_train)
y_pred = classifier_knn.predict(X_test)
preds = classifier_knn.predict(sample)
pred_species=list()
for p in preds:
    pred_species.append(response_vector[p])
print(pred_species)
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
plt.scatter(X_test[0:20],X_train[0:20])
plt.show()
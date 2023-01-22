from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer 
from lazypredict.Supervised import LazyClassifier

data=load_breast_cancer()
X=data.data
y=data.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5,random_state=123)
clf=LazyClassifier(verbose=0,ignore_warnings=True)
models,f=clf.fit(X_train,X_test,y_train,y_test)


print(models)

print(f)#same output as models
import pandas as pd
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
import joblib

iris = datasets.load_iris()
columns=["Petal length","Petal Width","Sepal Length","Sepal Width"]
x = pd.DataFrame(iris.data, columns=columns)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

model = svm.SVC(kernel='rbf')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy: %.3f" %model.score(X_test, y_pred))

joblib.dump(model, './server/src/svm_model.pkl')
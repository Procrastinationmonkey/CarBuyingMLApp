import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing
dataset = pd.read_csv('C:\Users\Dell\Desktop\Car_Buying\data\Buyingdata.csv')
X = dataset.iloc[:, [2, 3]].values #inputs(age,salary)
y = dataset.iloc[:, 4].values #column to predict(car bought/not)

# Splitting Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 30) #seed=30

# Scaling features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Random Forest Classification 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 30)
classifier.fit(X_train, y_train)

# Predicting the Test set results
prediction = classifier.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction)

# Visualising Test results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Accuracy of our Model
print("Accuracy of Model",classifier.score(X_test,y_test)*100,"%")


# Accuracy of our Model
print("Accuracy of Model",classifier.score(X_train,y_train)*100,"%")

#Saving to pickle
from sklearn.externals import joblib

RandomForestModel = open("C:\Users\Dell\Desktop\Car_Buying\models\\randomforestmodel.pkl","wb")

joblib.dump(classifier,RandomForestModel)

RandomForestModel.close()
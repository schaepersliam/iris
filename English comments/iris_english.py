import pandas as pd
from sklearn.svm import SVC
from sklearn import model_selection

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

#Features and labels
array = dataset.values
#X are the features
X = array[:,0:4]
#Y are the labels for the features
Y = array[:,4]

#Spliting into train and test features and labels
seed = 7
test_size = 0.20
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

#Defining the classifier
clf = SVC(kernel='linear')
clf.fit(features_train, labels_train)
accuacy = clf.score(features_test, labels_test)
print(accuacy)

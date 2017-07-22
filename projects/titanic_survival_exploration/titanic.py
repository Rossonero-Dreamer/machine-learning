import pandas as pd
import numpy as np

#load raw data
in_file = "titanic_data.csv"
data  = pd.read_csv(in_file, sep=',')

#extract certain features
outcomes = data['Survived']
features = ['Survived','Name','Ticket','Cabin','Embarked','PassengerId']
data = data.drop(features, axis=1)

#preprocessing
data['Sex'] = data['Sex'].apply(lambda x:1. if x=='female' else 0.)
age_mean = data['Age'].mean()
data = data.fillna(value=age_mean)

#split dataset
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=450)
for train_index, test_index in sss.split(data,outcomes):
    X_train = data.iloc[train_index]
    X_test = data.iloc[test_index]
    y_train = outcomes.iloc[train_index]
    y_test = outcomes.iloc[test_index]

#use decision tree classifier
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

#evaluate test results
from sklearn.metrics import accuracy_score
print("Accuracy Score: ", accuracy_score(y_test, pred))

#output the tree
import pydotplus
dot_data = tree.export_graphviz(clf, out_file=None,
                               feature_names=data.columns,
                               class_names=['Perished','Survived'],
                               filled=True, rounded=True,
                               proportion=True,
                               special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("titanic_tree.pdf")

# 1 Importing the libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# 2 Importing the dataset
# dataset = pd.read_csv(
#     'AIO-2024/module3-week3-decision_tree/data/age_likes_raise_salary.csv')

# X = dataset[['Age', 'Likes English', 'Likes AI']].values
# y = dataset[['Raise Salary']].values.reshape(-1,)

# # define classifier
# clf = DecisionTreeClassifier(max_depth=3)

# # train
# clf = clf.fit(X, y)


# # visualization
# plt.figure(figsize=(20, 10))
# plot_tree(clf, feature_names=['Age', 'Likes English', 'Likes AI'],
#           class_names=['No', 'Yes'], filled=True)
# plt.show()

iris_X, iris_y = sklearn.datasets.load_iris(return_X_y=True)


X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_y,
    test_size=0.2,
    random_state=42)

dt_classifier = DecisionTreeClassifier()

dt_classifier.fit(X_train, y_train)

y_pred = dt_classifier.predict(X_test)
print(y_pred)
accuracy_score(y_test, y_pred)

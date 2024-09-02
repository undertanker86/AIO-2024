#Question 3:

# import numpy as np
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# # Load the diabetes dataset
# iris_X, iris_y = datasets.load_iris(return_X_y=True)
# # Split train:test = 8:2
# X_train, X_test, y_train, y_test = train_test_split(
# iris_X,
# iris_y,
# test_size=0.2,
# random_state=42
# )
# # Scale the features using StandardScaler
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# # Build KNN Classifier
# knn_classifier = KNeighborsClassifier(n_neighbors=5)
# knn_classifier.fit(X_train, y_train)
# # Predict and Evaluate test set
# y_pred = knn_classifier.predict(X_test)
# accuracy_score(y_test, y_pred)

#Question 5:

# import numpy as np
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsRegressor


# # Load the diabetes dataset
# diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
# # Split train:test = 8:2
# X_train, X_test, y_train, y_test = train_test_split(
# diabetes_X,
# diabetes_y,
# test_size=0.2,
# random_state=42
# )

# # Scale the features using StandardScaler
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Build KNN model
# knn_regressor = KNeighborsRegressor(n_neighbors=5)
# knn_regressor.fit(X_train, y_train)

#Question 7:
# Import library
# import numpy as np
# from datasets import load_dataset
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_extraction.text import CountVectorizer
# # Load IMDB dataset
# imdb = load_dataset("imdb")
# imdb_train, imdb_test = imdb['train'], imdb['test']
# # Convert text to vector using BoW
# *** Your code here ***
# vectorizer = CountVectorizer(max_features=1000)
# X_train = vectorizer.fit_transform(imdb_train[’text’]).toarray()
# X_test = vectorizer.transform(imdb_test[’text’]).toarray()
# y_train = np.array(imdb_train[’label’])
# y_test = np.array(imdb_test[’label’])
# # Scale the features using StandardScaler
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# # Build KNN Classifier
# knn_classifier = KNeighborsClassifier(n_neighbors=1, algorithm=’ball_tree’)
# knn_classifier.fit(X_train, y_train)
# # predict test set and evaluate
# y_pred = knn_classifier.predict(X_test)


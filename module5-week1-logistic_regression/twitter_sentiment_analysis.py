import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from nltk.tokenize import TweetTokenizer

from collections import defaultdict
def text_normalize(text):
    # Retweet old acronym "RT" removal
    text = re.sub(r'^RT[\s]+', '', text)
    
    # Hyperlinks removal
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    
    # Hashtags removal
    text = re.sub(r'#', '', text)
    
    # Punctuation removal
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization
    tokenizer = TweetTokenizer(
        preserve_case=False,
        strip_handles=True,
        reduce_len=True
    )
    text_tokens = tokenizer.tokenize(text)
    
    return text_tokens

def get_freqs(df):
    freqs = defaultdict(lambda: 0)
    for idx, row in df.iterrows():
        tweet = row['tweet']
        label = row['label']
        
        tokens = text_normalize(tweet)
        for token in tokens:
            pair = (token, label)
            freqs[pair] += 1
    
    return freqs

def get_feature(text, freqs):
    tokens = text_normalize(text)

    X = np.zeros(3)
    X[0] = 1

    for token in tokens:
        X[1] += freqs[(token, 0)]
        X[2] += freqs[(token, 1)]

    return X

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y_hat, y):
    y_hat = np.clip(y_hat, 1e-7, 1 - 1e-7)
    return (-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)).mean()

def predict(X, theta):
    dot_product = np.dot(X, theta)
    y_hat = sigmoid(dot_product)
    return y_hat

def compute_gradient(X, y, y_hat):
    return np.dot(X.T, (y_hat - y)) / y.size

def update_theta(theta, gradient, lr):
    return theta - lr * gradient

def compute_accuracy(X, y, theta):
    y_hat = predict(X, theta).round()
    acc = (y_hat == y).mean()
    return acc


if __name__ == '__main__':
    df = pd.read_csv('D:/AIO-2024-WORK/AIO-2024/module5-week1-logistic_regression/data/sentiment_analysis.csv', index_col ='id')
    
    X = []
    y = []

    freqs = get_freqs(df)
    for idx , row in df.iterrows():
        tweet = row['tweet']
        label = row['label']
        X_i = get_feature (tweet , freqs )

        X.append(X_i)
        y.append(label)
    X = np.array(X)
    y = np.array(y)

    val_size = 0.2
    test_size = 0.125
    random_state = 2
    is_shuffle = True

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_size,
        random_state=random_state,
        shuffle=is_shuffle
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train,
        test_size=test_size,
        random_state=random_state,
        shuffle=is_shuffle
    )

    normalizer = StandardScaler()

    X_train[:, 1:] = normalizer.fit_transform(X_train[:, 1:])
    X_val[:, 1:] = normalizer.transform(X_val[:, 1:])
    X_test[:, 1:] = normalizer.transform(X_test[:, 1:])

    lr = 0.01
    epochs = 200
    batch_size = 128

    np.random.seed(0)
    theta = np.random.uniform(size=X_train.shape[1])

    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []

    for epoch in range(epochs):
        train_batch_losses = []
        train_batch_accs = []
        val_batch_losses = []
        val_batch_accs = []

        for i in range(0, X_train.shape[0], batch_size):
            X_i = X_train[i:i+batch_size]
            y_i = y_train[i:i+batch_size]

            y_hat = predict(X_i, theta)
            train_loss = compute_loss(y_hat, y_i)
            gradient = compute_gradient(X_i, y_i, y_hat)
            theta = update_theta(theta, gradient, lr)

            train_batch_losses.append(train_loss)
            train_acc = compute_accuracy(X_train, y_train, theta)
            train_batch_accs.append(train_acc)

        y_val_hat = predict(X_val, theta)
        val_loss = compute_loss(y_val_hat, y_val)
        val_batch_losses.append(val_loss)

        val_acc = compute_accuracy(X_val, y_val, theta)
        val_batch_accs.append(val_acc)

        train_batch_loss = sum(train_batch_losses) / len(train_batch_losses)
        val_batch_loss = sum(val_batch_losses) / len(val_batch_losses)
        train_batch_acc = sum(train_batch_accs) / len(train_batch_accs)
        val_batch_acc = sum(val_batch_accs) / len(val_batch_accs)

        train_losses.append(train_batch_loss)
        val_losses.append(val_batch_loss)
        train_accs.append(train_batch_acc)
        val_accs.append(val_batch_acc)

        print(f'EPOCH {epoch + 1}:\tTraining loss: {train_batch_loss:.3f}\tValidation loss: {val_batch_loss:.3f}')
    val_set_acc = compute_accuracy(X_val, y_val, theta)
    test_set_acc = compute_accuracy(X_test, y_test, theta)

    print('Evaluation on validation and test set:')
    print(f'Accuracy on validation set: {val_set_acc}')
    print(f'Accuracy on test set: {test_set_acc}')

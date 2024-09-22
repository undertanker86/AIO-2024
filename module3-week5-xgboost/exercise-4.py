import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


if __name__ == "__main__":
    dataset_path = 'AIO-2024/module3-week5-xgboost/Problem4.csv'
    data_df = pd.read_csv(dataset_path)
    print(data_df.head())
    X, y = data_df.iloc[:, : -1], data_df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=7)
    xg_class = xgb.XGBClassifier(seed=7)
    xg_class . fit(X_train, y_train)
    preds = xg_class.predict(X_test)

    train_acc = accuracy_score(y_train, xg_class . predict(X_train))
    test_acc = accuracy_score(y_test, preds)
    print(f'Train ACC: { train_acc }')
    print(f'Test ACC: { test_acc }')

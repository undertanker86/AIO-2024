import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


if __name__ == "__main__":
    # Load the data
    data = pd.read_csv('AIO-2024/module3-week3-ensemble_learning/Housing.csv')
    categorical_cols = data.select_dtypes(
        include=['object']).columns.tolist()  # Object data type columns
    ordinal_encoder = OrdinalEncoder()
    encoded_categorical_cols = ordinal_encoder.fit_transform(
        data[categorical_cols])  # 545,7

    encoded_categorical_df = pd.DataFrame(
        encoded_categorical_cols, columns=categorical_cols)  # Fill data encoded data into a dataframe have nname columns as categorical columns
    # Drop the categorical columns from the data
    numerical_df = data.drop(columns=categorical_cols)
    # Concatenate the numerical and encoded categorical columns
    ecoded_df = pd.concat([numerical_df, encoded_categorical_df], axis=1)
    normalizer = StandardScaler()
    dataset_arr = normalizer.fit_transform(ecoded_df)  # Normalize the data
    X, y = dataset_arr[:, 1:], dataset_arr[:, 0]  # Split the data

    test_size = 0.3
    random_state = 1
    is_shuffle = True
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=is_shuffle)

    # Random Forest Regressor
    regressor = RandomForestRegressor(
        random_state=random_state
    )
    regressor .fit(X_train, y_train)

    # AdaBoost Regressor
    # regressor = AdaBoostRegressor(random_state=random_state)

    # Gradient Boosting Regressor
    # regressor = GradientBoostingRegressor(random_state=random_state)

    y_pred = regressor.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    print(f'Mean Absolute Error : { mae}')
    print(f'Mean Squared Error : { mse}')

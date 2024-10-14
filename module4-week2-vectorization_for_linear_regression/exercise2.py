import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def predict(X, w, b):
    return np.dot(X, w) + b

def gradient(y_hat, y, x):
    loss = y_hat - y
    dw = x.T.dot(loss) / len(y)
    db = np.sum(loss) / len(y)
    cost = np.sum( loss **2) / (2* len(y))
    return dw, db, cost

def update_weight(w, b, lr, dw, db):
    w_new = w - dw*lr
    b_new = b - db*lr
    return w_new, b_new

def linear_regression_vectorized (X , y , learning_rate =0.01 , num_iterations =200):
    n_samples , n_features = X.shape
    w = np.zeros( n_features )
    b = 0 # Initialize bias
    losses = []
    for _ in range (num_iterations):
        y_hat = predict(X , w , b )
        dw , db , cost = gradient( y_hat , y , X )
        w , b = update_weight(w , b , learning_rate , dw , db )
        losses.append(cost)
    return w , b , losses




if __name__ == '__main__':
#1
    # Read the CSV file
    df = pd.read_csv('AIO-2024/module4-week2-vectorization_for_linear_regression/data/BTC-Daily.csv')
    
    # Remove duplicates
    df = df.drop_duplicates()

    df['date'] = pd.to_datetime(df['date'])
    
    df['year'] = df['date'].dt.year

    date_range = str(df['date'].dt.date.min()) + ' to ' + str(df['date'].dt.date.max())
    print(f"Date Range: {date_range}")

    unique_years = list({i for i in df['date'].dt.year})
    # print(len(unique_years))
    # Plot the data for each year
    # for year in unique_years:
    #     year_data = df[df['year'] == year]  # Filter data for the year
        
    #     # Plot the data
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(year_data['date'], year_data['close'], label='Close Price')
    #     plt.title(f'Bitcoin Closing Prices - {year}')
    #     plt.xlabel('Date')
    #     plt.ylabel('Closing Price (USD)')
    #     plt.xticks(rotation=45)
    #     plt.tight_layout()
    #     plt.show()
# 2
    # Filter data for 2019-2022
    # df_filtered = df[(df['date'] >= '2019-01-01') & (df['date'] <= '2022-12-31')]

    # # Convert date to matplotlib format
    # df_filtered['date'] = df_filtered['date'].map(mdates.date2num)

    # # Create the candlestick chart
    # fig, ax = plt.subplots(figsize=(20, 6))

    # candlestick_ohlc(ax, df_filtered[['date', 'open', 'high', 'low', 'close']].values
    #     , width=0.6, colorup='g', colordown='r')

    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # fig.autofmt_xdate()

    # plt.title('Bitcoin Candlestick Chart (2019-2022)')
    # plt.xlabel('Date')
    # plt.ylabel('Price (USD)')
    # plt.grid(True)

    # # Save the plot as a PDF
    # plt.savefig('bitcoin_candlestick_2019_2022.pdf')

    # plt.show()
# 3
    scalar = StandardScaler()
    df["Standardized_Close_Price"] = scalar.fit_transform(df["close"].values.reshape(-1, 1))
    df["Standardized_Open_Price"] = scalar.fit_transform(df["open"].values.reshape(-1, 1))
    df["Standardized_High_Price"] = scalar.fit_transform(df["high"].values.reshape(-1, 1))
    df["Standardized_Low_Price"] = scalar.fit_transform(df["low"].values.reshape(-1, 1))
    df['date_str'] = df['date'].dt.strftime('%Y%m%d%H%M%S')
    df['NumericaDate'] = pd.to_numeric(df['date_str'])

    df.drop(columns=['date_str'], inplace=True)

    X = df[['Standardized_Open_Price', 'Standardized_High_Price', 'Standardized_Low_Price']]
    y = df['Standardized_Close_Price']
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size =0.3 ,random_state =42 , shuffle = True )
    w , b , losses = linear_regression_vectorized( X_train.values , y_train.values , learning_rate =0.01 , num_iterations =200)
    # plt.plot( losses )
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.title('Loss')
    # plt.show()

    # Make predictions on the test set
    y_pred = predict(X_test, w, b)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

    # Calculate MAE
    mae = np.mean(np.abs(y_pred - y_test))

    # Calculate MAPE
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Calculate R-squared on training data
    y_train_pred = predict(X_train, w, b)
    train_accuracy = r2_score(y_train, y_train_pred)

    # Calculate R-squared on testing data
    test_accuracy = r2_score(y_test, y_pred)

    print("Root Mean Square Error (RMSE):", round(rmse, 4))
    print("Mean Absolute Error (MAE):", round(mae, 4))
    print("Training Accuracy (R-squared):", round(train_accuracy, 4))
    print("Testing Accuracy (R-squared):", round(test_accuracy, 4))
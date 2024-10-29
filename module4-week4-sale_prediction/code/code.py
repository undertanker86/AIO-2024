import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def r2score_test( y_pred , y ):
    rss = np.sum(( y_pred - y ) ** 2)
    tss = np.sum(( y - y.mean() ) ** 2)
  
    r2 = 1 - (rss / tss)
    return r2



if __name__ == '__main__':



    df = pd.get_dummies(pd.read_csv('D:\AIO-2024-WORK\AIO-2024\module4-week4-sale_prediction\data\SalesPrediction.csv'))
    df = df.fillna(df.mean())

    # Get features
    X = df[['TV', 'Radio', 'Social Media', 'Influencer_Macro', 'Influencer_Mega', 'Influencer_Micro', 'Influencer_Nano']]
    y = df[['Sales']]

    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=0)

    scaler = StandardScaler ()
    X_train_processed = scaler.fit_transform( X_train )
    X_test_processed = scaler.fit_transform( X_test )
    print(scaler.mean_[0])

    poly_features = PolynomialFeatures (degree =2)
    X_train_poly = poly_features.fit_transform( X_train_processed )
    X_test_poly = poly_features.transform(X_test_processed)
    poly_model = LinearRegression()
    poly_model.fit( X_train_poly , y_train )
    preds = poly_model.predict( X_test_poly )
    print(r2_score( y_test , preds ))

    # y_pred = np.array([1 , 2 , 3 , 4 , 5])
    # y = np.array([1 , 2 , 3 , 4 , 5])
    # y_pred = np.array([1 , 2 , 3 , 4 , 5])
    # y = np.array([3 , 5 , 5 , 2 , 4])
    # print(r2score_test(y_pred , y))
    # print(r2_score(y , y_pred))




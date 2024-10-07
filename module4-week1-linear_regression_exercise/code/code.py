import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def get_column(data, index):
    result = data[:, index]
    return result
def prepare_data_2( file_name_dataset ) :
    data = np.genfromtxt(file_name_dataset, delimiter=',', skip_header=1)

    tv_data = get_column(data, 0)
    radio_data = get_column(data, 1)
    newspaper_data = get_column(data, 2)
    sales_data = get_column( data , 3)

    X = [[1 , x1 , x2 , x3 ] for x1 , x2 , x3 in zip( tv_data , radio_data , newspaper_data )]
    y = sales_data
    return X , y

def initialize_params_2():
    bias = 0
    w1 = random.gauss( mu =0.0 , sigma =0.01)
    w2 = random.gauss( mu =0.0 , sigma =0.01)
    w3 = random.gauss( mu =0.0 , sigma =0.01)
    return [0 , -0.01268850433497871 , 0.004752496982185252 , 0.0073796171538643845]

def prepare_data(file_name_dataset):
    data = np.genfromtxt(file_name_dataset, delimiter=',', skip_header=1)


    N = len(data)

    tv_data = get_column(data, 0)
    radio_data = get_column(data, 1)
    newspaper_data = get_column(data, 2)
    sales_data = get_column( data , 3)

    X = [tv_data, radio_data, newspaper_data]
    y = sales_data

    return X, y


def initialize_params():
    w1 = 0.016992259082509283
    w2 = 0.0070783670518262355
    w3 = -0.002307860847821344
    b = 0.0

    return w1, w2, w3, b

def predict(x1, x2, x3, w1, w2, w3, b):
    y_hat = w1 * x1 + w2 * x2 + w3 * x3 + b
    return y_hat

def predict_2(X_features, weight):
    y_hat = np.dot(X_features, weight)
    return y_hat

def compute_loss_mse(y, y_hat):
    loss = (y - y_hat) ** 2
    return loss



def compute_gradient_w(X_features, y, y_hat):
    gradient = -2 * np.dot(X_features, (y - y_hat))
    return gradient


def compute_loss_mae(y, y_hat):
    loss = abs(y - y_hat)
    return loss

def compute_gradient_wi(xi, y, y_hat):
    gradient = -2 * xi * (y - y_hat)
    return gradient

def compute_gradient_b(y, y_hat):
    gradient = -2 * (y - y_hat)
    return gradient

def update_weight_wi(wi, dl_dwi, lr):
    wi = wi - lr * dl_dwi
    return wi


def update_weight_b(b, dl_db, lr):
    b = b - lr * dl_db
    return b

def update_weight( weights , dl_dweights , lr ):
    weights = weights - lr * dl_dweights
    return weights

def implement_linear_regression(X_data, y_data, epoch_max = 50, lr = 1e-5):
    losses = []


    w1, w2, w3, b = initialize_params()

    N= len(y_data)

    for epoch in range(epoch_max):
        for i in range(N):
            x1 = X_data[0][i]
            x2 = X_data[1][i]
            x3 = X_data[2][i]


            y = y_data[i]

            y_hat = predict( x1 , x2 , x3 , w1 , w2 , w3 , b )

            loss = compute_loss_mse(y , y_hat)
            dl_dw1 = compute_gradient_wi( x1 , y , y_hat )
            dl_dw2 = compute_gradient_wi( x2 , y , y_hat )
            dl_dw3 = compute_gradient_wi( x3 , y , y_hat )
            dl_db = compute_gradient_b(y , y_hat )

            w1 = update_weight_wi( w1 , dl_dw1 , lr )
            w2 = update_weight_wi( w2 , dl_dw2 , lr )
            w3 = update_weight_wi( w3 , dl_dw3 , lr )
            b = update_weight_b(b , dl_db , lr )
            losses.append( loss )
    return w1 , w2 , w3 ,b , losses

def implement_linear_regression_nsamples(X_data, y_data, epoch_max=50, lr=1e-5):
    losses = []
    w1 , w2 , w3 , b = initialize_params()
    N = len( y_data )

    for epoch in range( epoch_max ) :
        loss_total = 0.0
        dw1_total = 0.0
        dw2_total = 0.0
        dw3_total = 0.0
        db_total = 0.0

        for i in range ( N ) :
            x1 = X_data [0][i]
            x2 = X_data [1][i]
            x3 = X_data [2][i]
            y = y_data[i]
            y_hat = predict( x1 , x2 , x3 , w1 , w2 , w3 , b )
            loss = compute_loss_mse(y , y_hat )
            dl_dw1 = compute_gradient_wi( x1 , y , y_hat )
            dl_dw2 = compute_gradient_wi( x2 , y , y_hat )
            dl_dw3 = compute_gradient_wi( x3 , y , y_hat )
            dl_db = compute_gradient_b( y , y_hat )
            loss_total += loss
            dw1_total += dl_dw1
            dw2_total += dl_dw2
            dw3_total += dl_dw3
            db_total += dl_db

        w1 = update_weight_wi( w1 , dw1_total / N , lr )
        w2 = update_weight_wi( w2 , dw2_total / N , lr )
        w3 = update_weight_wi( w3 , dw3_total / N , lr )
        b = update_weight_b( b , db_total / N , lr )
        losses.append( loss_total / N )

    return w1 , w2 , w3 , b , losses 

def implement_linear_regression(X_feature, y_ouput, epoch_max=50, lr=1e-5):
    losses = []
    weights = initialize_params()
    N = len(y_ouput)
    for epoch in range(epoch_max):
        print("epoch", epoch)
        for i in range(N):
            # get a sample - row i
            features_i = X_feature[i]
            # y = sales_data[i]
            y = y_ouput[i]

            # compute output
            y_hat = predict_2(features_i, weights)

            # compute loss
            loss = compute_loss_mse(y, y_hat)

            # compute gradient w1, w2, w3, b
            dl_dweights = compute_gradient_w(features_i, y, y_hat)

            # update parameters
            weights = update_weight(weights, dl_dweights, lr)

            # logging
            losses.append(loss)
    return weights, losses


if __name__ == "__main__":
    #Task 1:
    # X, y = prepare_data('AIO-2024/module4-week1-linear_regression_exercise/data/advertising.csv')
    # list = [ sum( X [0][:5]) , sum( X [1][:5]) , sum( X [2][:5]) , sum( y [:5]) ]
    # print(list)

    # Task 2:
    # ( w1 , w2 , w3 ,b , losses ) = implement_linear_regression (X , y )

    # plt.plot( losses [:100])
    # plt.xlabel("# iteration ")
    # plt.ylabel(" Loss ")
    # plt.show()

    # ( w1 , w2 , w3 ,b , losses ) = implement_linear_regression (X , y )
    # print ( w1 , w2 , w3 )

    # tv = 19.2
    # radio = 35.9
    # newspaper = 51.3
    # w1 , w2 , w3 ,b , losses = implement_linear_regression(X , y , epoch_max =50 , lr =1e-5)
    # sales = predict( tv , radio , newspaper , w1 , w2 , w3 , b )
    # print (f'predicted sales is { sales }')

    # l = compute_loss_mae ( y_hat =1 , y =0.5)
    # print( l )

    # Task 3:
    # w1 , w2 , w3 ,b , losses = implement_linear_regression_nsamples( X , y , epoch_max =1000 ,lr =1e-5)
    # print( losses )
    # print( w1, w2, w3)
    # plt.plot(losses)
    # plt.xlabel("# epoch ")
    # plt.ylabel("MSE Loss ")
    # plt.show()

    #Task 4:
    X, y = prepare_data_2('AIO-2024/module4-week1-linear_regression_exercise/data/advertising.csv')
    W , L = implement_linear_regression(X , y , epoch_max =50 , lr =1e-5)
    print (L[9999])
    # plt.plot(L[0:100])
    # plt.xlabel("#iteration")
    # plt.ylabel("Loss")
    # plt.show()
    



    
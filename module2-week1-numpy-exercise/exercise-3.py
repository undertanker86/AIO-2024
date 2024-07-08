import numpy as np
import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv('advertising.csv')
    data = df.to_numpy()
    # Question 15
    print(data.max(axis=0))
    print(data.argmax(axis=0))
    # Question 16
    print(np.mean(data[:, 0]))
    # Question 17
    records_sale = np.sum(data[:, -1] >= 20)
    # Question 18
    mean_radio = np.mean(data[data[:, -1] >= 15, 1])
    # Question 19
    mean_newspaper = np.mean(data[:, 2])
    sum_sales = np.sum(data[data[:, 2] > mean_newspaper, -1])
    print(sum_sales)
    #  Question 20
    mean_sales = np.mean(data[:, -1])
    print(mean_sales)
    scores = np.array(["Good" if sale > mean_sales else "Average" if sale ==
                    mean_sales else "Bad" for sale in data[:, -1]])
    print(scores[7:10])

    # Question 21
    sort_data_by_sales = data[data[:, -1].argsort()]

    after_mean_sales = sort_data_by_sales[sort_data_by_sales[:, -1] > mean_sales]
    result_after = after_mean_sales[0]

    before_mean_sales = sort_data_by_sales[sort_data_by_sales[:, -1] <= mean_sales]
    result_before = before_mean_sales[-1]

    result_final = min(result_before[-1], result_after[-1])

    scores_2 = np.array(["Good" if sale > result_final else "Average" if sale ==
                        result_final else "Bad" for sale in data[:, -1]])
    print(scores_2[7:10])

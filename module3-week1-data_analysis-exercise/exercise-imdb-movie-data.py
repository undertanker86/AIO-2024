import pandas as pd


def rating_group(rating):
    if rating >= 7.5:
        return 'Good'
    elif rating >= 6.0:
        return 'Average'
    else:
        return 'Bad'


if __name__ == '__main__':
    # 1
    dataset_path = 'AIO-2024\module3-week1-data_analysis-exercise\IMDB-Movie-Data.csv'
    data = pd.read_csv(dataset_path)
    data_index = data.set_index('Title')

    # 2
    # print(data.head(5))

    # 3
    # print(data.info())
    # print(data.describe())

    # 4
    genre = data['Genre']
    genre_1 = data[['Genre']]
    some_cols = data[['Title', 'Genre', 'Actors', 'Director', 'Rating']]
    data.iloc[10:15][['Title', 'Rating', 'Revenue (Millions)']]

    # 5
    # print(data[((data['Year'] >= 2010) & (data['Year'] <= 2015)) & (data['Rating'] < 6.0) & (data['Revenue (Millions)'] > data['Revenue (Millions)'].quantile(0.95))])

    # 6
    # print(data.groupby('Director')[['Rating']].mean().head(5))

    # 7
    # print(data.groupby('Director')[['Rating']].mean().sort_values(by='Rating', ascending=False).head(5))
    # 8
    print(data.isnull().sum())

    # 9
    # data.drop('Metescore', axis=1).head()
    # data.dropna()

    # 10
    # revenue_mean = data_index['Revenue (Millions)'].mean()
    # data['Revenue (Millions)'].fillna(revenue_mean, inplace=True)

    # 11
    data['Rating_category'] = data['Rating'].apply(rating_group)
    print(data.head(5))

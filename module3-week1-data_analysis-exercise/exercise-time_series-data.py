import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
if __name__ == '__main__':
    # 1
    dataset_path = 'AIO-2024\module3-week1-data_analysis-exercise\opsd_germany_daily.csv'
    opsd_daily = pd.read_csv(dataset_path, index_col='Date', parse_dates=True)
    opsd_daily['Year'] = opsd_daily.index.year
    opsd_daily['Month'] = opsd_daily.index.month
    opsd_daily['Weekday Name'] = opsd_daily.index.day_name()

    opsd_daily.sample(5, random_state=0)

    # 2
    print(opsd_daily.loc['2014-01-20':'2014-01-22'])

    # 3
    # sns.set(rc={'figure.figsize': (11, 4)})
    # opsd_daily['Consumption'].plot(linewidth=0.5)
    # plt.show()

    # 4
    # fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    # for name, ax in zip(['Consumption', 'Solar', 'Wind'], axes):
    #     sns.boxplot(data=opsd_daily, x='Month', y=name, ax=ax)
    #     ax.set_ylabel('GWh')
    #     ax.set_title(name)

    #     if ax != axes[-1]:
    #         ax.set_xlabel('')
    # plt.show()

    # 5
    # print(pd.date_range('1998-03-10', '1998-03-15', freq='D'))
    # print(pd.date_range('1998-03-10', periods=8, freq='h'))
    # times_sample = pd.to_datetime (['2013 -02 -03', '2013 -02 -06', '2013 -02 -08'])
    # consum_sample = opsd_daily.loc[ times_sample , ['Consumption']].copy()

    # consum_freq = consum_sample.asfreq('D')

    # consum_freq ['Consumption - Forward Fill'] = consum_sample.asfreq('D', method ='ffill')

    # 6
    data_columns = ['Consumption', 'Wind', 'Solar', 'Wind+Solar']
    opsd_weekly_mean = opsd_daily[data_columns].resample('W').mean()

    # 7
    opsd_daily[data_columns].rolling(7, center=True).mean()

    # 8
    # opsd_7d = opsd_daily[data_columns].rolling(7, center=True).mean()

    # opsd_365d = opsd_daily[data_columns].rolling(
    #     window=365, center=True, min_periods=360).mean()

    # fig, ax = plt.subplots()
    # ax.plot(opsd_daily['Consumption'], marker='.', markersize=2,
    #         color='0.6', linestyle='None', label='Daily')
    # ax.plot(opsd_7d['Consumption'], linewidth=2, label='7-d Rolling Mean')
    # ax.plot(opsd_365d['Consumption'], color='0.2',
    #         linewidth=3, label='Trend (365-d Rolling Mean)')

    # # Set x-ticks to yearly interval and add legend and labels
    # ax.xaxis.set_major_locator(mdates.YearLocator())
    # ax.legend()
    # ax.set_xlabel('Year')
    # ax.set_ylabel('Consumption (GWh)')
    # ax.set_title('Trends in Electricity Consumption')
    # plt.show()

    # Plot 365-day rolling mean time series of wind and solar power

    # fig, ax = plt.subplots()
    # for nm in ['Wind', 'Solar', 'Wind+Solar']:
    #     ax.plot(opsd_365d[nm], label=nm)

    # # Set x-ticks to yearly interval, adjust y-axis limits, add legend and labels
    # ax.xaxis.set_major_locator(mdates.YearLocator())
    # ax.set_ylim(0, 400)
    # ax.legend()
    # ax.set_ylabel('Production (GWh)')
    # ax.set_title('Trends in Electricity Production (365-d Rolling Means)')
    # plt.show()

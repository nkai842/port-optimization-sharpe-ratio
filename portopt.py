import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import csv
from datetime import datetime
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

today = datetime.today().strftime('%Y-%m-%d')


weights = np.array([])

tickers = []
filename = "equities.csv"
try:
    with open(filename, 'r') as myCSV:
        data = csv.reader(myCSV)
        next(data)
        for row in data:
            tickers.append(row)

        # rest of processing goes here
    myCSV.close()

except FileNotFoundError:
    print('no file!')

df = pd.DataFrame()
for tick in tickers:
    try:
        df[tick[0]] = web.DataReader(tick[0],data_source='yahoo',start="2009-01-01" , end=today)['Adj Close']
        print(tick[0], "added")
    except:
        print(tick[0], "not added")
        pass

mu = expected_returns.mean_historical_return(df)#returns.mean() * 252
S = risk_models.sample_cov(df) #Get the sample covariance matrix
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe() #Maximize the Sharpe ratio, and get the raw weights
cleaned_weights = ef.clean_weights()
print(cleaned_weights) #Note the weights may have some rounding error, meaning they may not add up exactly to 1 but should be close
ef.portfolio_performance(verbose=True)


latest_prices = get_latest_prices(df)
weights = cleaned_weights
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=2000)
allocation, leftover = da.lp_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))

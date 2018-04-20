
# coding: utf-8

# # Chocolate confectionery

# This report briefly addresses forecasting of Monthly production of chocolate confectionery in Australia: tonnes. July 1957 – Aug 1995 (see [here](https://datamarket.com/data/set/22rl/monthly-production-of-chocolate-confectionery-in-australia-tonnes-july-1957-aug-1995#!ds=22rl&display=line)) using Python.

# ## 1. Data understanding

import warnings
def ignore_warnings(*args, **kwargs):
    pass
warnings.warn = ignore_warnings 

from math import sqrt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

from pyramid.arima import auto_arima

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


# The first step is to load the dataset. In order to read the dataset as time series, a date parser function is defined as follows.  :

#Load dataset
df = pd.read_csv("chocolate.csv", index_col = 'Month', parse_dates = ['Month'], 
                 date_parser = lambda x : pd.to_datetime(x, format = '%Y-%m'))

df.index


# Note that the type of the index variable, i.e., Month, is not of object type and converted into datetime. Now that, we have a sneak peek at the chocolate confectionary dataset.

print(df.head())


# Then, the descriptive summary of the dataset is provided as follows. 

print(df.describe())


# ## 2. Exploring data

#Set up helper function for data visualization 
def plt_(dataset, title):    
    plt.figure(figsize=(12,6))
    plt.plot(dataset, color = 'b')
    plt.ylabel('Tonnes')
    plt.title(title)
    plt.show()
    
def density_plt_(dataset):
    plt.figure(figsize=(10,5))
    sns.distplot(dataset)
    plt.title('Density plot')
    plt.show()   


#Time series plot
plt_(df, "Monthly production of chocolate confectionery")


# The monthly production plot suggests that there is an increasing trend of sales over time. Further, it illustrates seasonality patterns over time which does not seem to be multiplicative. Another observation is that there is not any obvious outlier in the dataset. Given the exisitng trend and seasonality, the dataset is therefore non-stationary. 

#Density plot
density_plt_(df)


# As illustrated by density plot, the observation represents a skewed distribution which can be roughly estimated by normal one. The underlying time series dataset can be decomposed into its componenets that are trend, seasonality, and sesidual as shown below. 

#Decomposition
def _decom(dataset):
      
    decomposition = seasonal_decompose(dataset)

    fig = plt.figure(figsize = (20, 10))
    ax = fig.add_subplot(411)
    ax.plot(dataset, label='Original', color = 'b')
    ax.legend(loc='best')
    ax = fig.add_subplot(412)
    ax.plot(decomposition.trend, label='Trend', color = 'b')
    ax.legend(loc='best')
    ax = fig.add_subplot(413)
    ax.plot(decomposition.seasonal,label='Seasonality', color = 'b')
    ax.legend(loc='best')
    ax= fig.add_subplot(414)
    ax.plot(decomposition.resid, label='Residuals', color = 'b')
    ax.legend(loc='best')
    plt.show()
    
_decom(df[df.columns[0]])


# As expected, increasing trend and seasonality signal are present in the dataset. It can further be validated by applying "Dickey-Fuller test" as follows.  

#Checking if the data is stationary or not
def stationarity_test(dataset):
        
    plt.figure(figsize = (10, 5))
    ax = plt.subplot()
    ax.set_xlabel("Year")
    ax.set_ylabel("Chocolate production (tonnes)")
    ax.plot(dataset, label = 'Original')
    ax.plot(pd.rolling_mean(dataset, window=12), label = 'Rolling Mean')
    ax.plot(pd.rolling_std(dataset, window=12), label = 'Rolling Std')
    ax.legend()
    plt.show()
    
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(dataset, autolag = 'AIC')
    dftest_output = pd.Series(dftest[0:4], index = ['Test statistic', 'p_value', 'Number of lags used', 'Number of observations used'])
    for keys, values in dftest[4].items():
        dftest_output['critical value (%s)' %keys] = values 
    return dftest_output

stationarity_test(df[df.columns[0]])


# The results of the statsitical test indicates the non-stationary the observations. Hence, the next step is to convert the dataset into a stationary time series. The first strategy to use is to apply "log" function on the observations which partially removes trend in the dataset as the result of the test indicates.


np.log(df[df.columns[0]]).head()



stationarity_test(np.log(df[df.columns[0]]))


# The next strategy to try is "differencing"


#Differencing: taking the differece with a particular time lag
ts_diff = np.log(df[df.columns[0]]) - np.log(df[df.columns[0]]).shift()
ts_diff.head()


ts_diff.dropna(inplace= True)
stationarity_test(ts_diff)


# Log transformation and differencing remove trend in teh dataset and convert it into a stationary one.  

# ## 3. Predictive model

# The dataset is now ready to fit the predictive model which is sasonal auto-regressive integrated moving averages (Seasonal ARIMA). In what follows, a class is defined to implement SARIMA along with a parameter tunning method which takes the advantage of [Pyramid](https://github.com/tgsmith61591/pyramid). 

class myARIMA(object):
    
    """
    Implementing Seasonal Auto-Regressive Integrated Moving Averages (SARIMA) method 
    
    """
    
    def __init__(self, timeseries, model):
        self.timeseries = timeseries
        self.model = model
        
    def cor_(self):
        """
        Constructing autocorrelation and partial autocorrelation plots 
        
        """
        _diff = self.timeseries - self.timeseries.shift()
        _diff.dropna(inplace=True)
        
        fig = plt.figure(figsize=(20,10)) 
        plt.subplot(211)
        plot_acf(_diff, ax=plt.gca())
        plt.title('Autocorrelation function')
        
        plt.subplot(212)
        plot_pacf(_diff, ax=plt.gca())
        plt.title('Partial autocorrelation function')
        
        plt.show()
        
    def auto_fit_(self, init_p, init_q, max_p, max_q, m, init_P, d, D, seasonal, stepwise):
        """
        Tunning the parameters of ARIMA 
        
        """
        
        #Creating train and test sets
        train_size = int(0.8*len(self.timeseries))
        train = self.timeseries[0:train_size]
        
        stepwise_fit = auto_arima(train, start_p=init_p, start_q=init_q, max_p=max_p, max_q=max_q, m=m,
                          start_P=init_P, seasonal=True, d=d, D=D, trace=True,
                          error_action='ignore',  # don't want to know if an order does not work
                          suppress_warnings=True,  # don't want convergence warnings
                          stepwise=stepwise)  # set to stepwise

        return stepwise_fit.summary()
               
    def model_(self, p, d, q, P, D, Q, m):
        """
        Building ARIMA model 
    
        """     
        #Creating train and test sets
        train_size = int(0.8*len(self.timeseries))
        train, test = self.timeseries[0:train_size], self.timeseries[train_size:]
        
        #Training data
        self.model = self.model(train, order=(p, d, q),seasonal_order=(P,D,Q,m))
        model_fit = self.model.fit()
        print(model_fit.summary())
        
        #Prediction
        y_hat = model_fit.predict(start=len(train), end=len(train) + (len(test)-1), typ='levels', dynamic=True) 
        error = sqrt(mean_squared_error(test, y_hat))
        print("Test RMSE: {:5.5f}".format(error))
        
        #Visualization
        self.__plot(np.exp(test), np.exp(y_hat))
        
        #Cross-validation
        self.__cross_validation(self.timeseries, self.model)
        
    def __plot(self, timeseries1, timeseries2):
        
        fig = plt.figure(figsize = (10,5))
        plt.subplot()
        plt.plot(timeseries1, color = 'b')
        plt.plot(timeseries2, color = 'g')
        plt.title('Forecasting plot')
        plt.show()
            
    def __cross_validation(self, timeseries, model):
        """
        Cross-validation 
    
        """         
        kf = TimeSeriesSplit(n_splits=3)
        err = []

        #Filtering train and test data
        for train_index, test_index in kf.split(self.timeseries):
            train_series = self.timeseries[train_index]
            test_series = self.timeseries[test_index]
            print('Number of observations: {}'.format((len(train_series) + len(test_series))))
            print('Training observations: {}'.format(len(train_series)))
            print('Testing observations: {}'.format(len(test_series)))
            cross_mod = self.model.fit(disp = 0)
            cross_pred = cross_mod.predict(start=len(train_series), end=len(train_series) + (len(test_series)-1), typ='levels', dynamic=True) 
            print("RMSE: {:5.5f}".format(sqrt(mean_squared_error(test_series, cross_pred))))
            err.append(sqrt(mean_squared_error(test_series, cross_pred)))
            
        print("Cross-Validation RMSE: {:5.5f}".format(np.mean(err)))     




obj = myARIMA(timeseries = np.log(df[df.columns[0]]), model = SARIMAX)




obj.cor_()


# The autocorrelation and partial autocorrelation graphs suggest to pick "2" as AR nad MA orders. In order to tune these parameters as well as seasonality ones, a stepwise search is performed using simulated "auto.arima" in Python. 



obj.auto_fit_(1, 1, 3, 3, 12, 0, 1, 1, seasonal = True, stepwise = True)


# The final step is to train the training set and then make forecasting using the test data.



obj.model_(3, 1, 3, 0, 1, 2, 12)


# The value of the cross-valdiation "RMSE" measure is quite close to that of train-test validation approach. It can be concluded that the SARIMA model fairly represents the time series dataset. Needless to say, the predictive model can further be improved to provide more accurate prediction. It can be done by further exploration of the dataset. 

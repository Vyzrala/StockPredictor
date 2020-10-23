from datetime import date
from random import randrange
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data
import seaborn as sb
import datetime
import numpy as np
import random
import time
import pickle
import os
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from keras import regularizers
import holidays
from matplotlib import dates
from scipy import stats


class Predictor:
    def __init__(self, correlation_threshold: float=0.75, split_ratio: float=0.8, backword_days: int=60,
                 epochs_number: int=200, batch: int=32, error_impact: float=0.8) -> None:
        """
            Description: 

            Parameters
            ----------

        """
        self.raw_dataset = None
        self.correlation_threshold = correlation_threshold
        self.train_data = None
        self.test_data = None
        self.split_ratio = split_ratio
        self.backword_days = backword_days
        self.epochs_number = epochs_number
        self.training_time = None
        self.batch = batch
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.RMSE = None
        self.error_distribution = None
        self.number_of_features = None
        self.error_impact = error_impact
        self.one_by_one_df = None

    def create_model(self, dataset: pd.DataFrame) -> None:
        """
            Description: 
                Method creates model, train it and provide error distribution

            Parameters
            ----------
            dataset : pandas DataFrame
                index: integers
                columns: Date, High, Low, Open, Close, Volume, Adj Close
        """
        self.raw_dataset = dataset.copy()
        
        # Additional features
        dataset['High_Low diff'] = dataset['High'] - dataset['Low']
        dataset['Open_Close diff'] = dataset['Open'] - dataset['Close']
        
        # Creating correlation matrix, extracting useful features for training
        correlation_matrix = dataset.corr()
        self.significant_features = list(
            correlation_matrix.loc[((correlation_matrix.Close >= self.correlation_threshold) |
                                    (correlation_matrix.Close <= -self.correlation_threshold)),
                                   ['Close']].index)
        self.number_of_features = len(self.significant_features)
        dataset = dataset[self.significant_features]

        # Splitting dataset into train and test sets
        dataset = np.array(dataset)
        split_index = int(dataset.shape[0] * self.split_ratio)
        self.train_data, self.test_data = dataset[:split_index, :].copy(), dataset[split_index:, :].copy()
        self.train_data = self.scaler.fit_transform(self.train_data)
        self.test_data = self.scaler.transform(self.test_data)
        x_train, y_train = self.get_xy_sets(self.train_data, self.backword_days)

        condidtion_1 = x_train is not None
        condidtion_2 = x_train is not np.array([])
        condidtion_3 = y_train is not None
        condidtion_4 = y_train is not np.array([])
        if not (condidtion_1 and condidtion_2 and condidtion_3 and condidtion_4):
            return
      
        # Model initialization
        input_shape = (self.backword_days, self.number_of_features)
        self.initialize_model(input_shape)

        # Model training
        start_time = time.time()
        self.model.fit(x_train, y_train, epochs=self.epochs_number, batch_size=self.batch, validation_split=.05)
        self.training_time = time.time() - start_time

        # Testing model on test set
        x_test, y_test = self.get_xy_sets(self.test_data, self.backword_days)
        y_predictions = None
        if x_test is None and y_test is None:
            return
        y_predictions = self.model.predict(x_test)

        # Model evaluation
        y_predictions = self.scaler.inverse_transform(y_predictions)
        y_test = self.scaler.inverse_transform(y_test)
        self.RMSE = pd.DataFrame([np.sqrt(np.mean((y_test[:,i] - y_predictions[:,i])**2)) for i in range(y_test.shape[1])],
                                 index=self.significant_features, columns=['RMSE [%]'])

        # Error distribution
        self.error_distribution = y_test - y_predictions
        self.error_distribution = self.error_distribution[(np.abs(stats.zscore(self.error_distribution))<3).all(axis=1)]

    def predict(self, days: int) -> pd.DataFrame: 
        """
            Description: 

            Parameters
            ----------

            
            Returns
            -------
            
        """
        begin_date = str(self.raw_dataset.Date.iloc[-1].date() + datetime.timedelta(days=1))

        if not self.model:
            raise Exception("Model have not been initilized")
        else:
            # Take last X (backword_days) days and unfilter unsignificant features
            input_set = np.array(self.raw_dataset[-self.backword_days:][self.significant_features])
            input_set = self.scaler.transform(input_set)
            input_set = input_set.reshape(1, self.backword_days, self.number_of_features)
            predictions = []

            day = 0
            while day < days:
                p = self.model.predict(input_set)
                p = self.scaler.inverse_transform(p)
                predictions.append(p)
                p += random.choice(self.error_distribution * self.error_impact)
                pe = self.scaler.transform(p)
                input_set = np.append(input_set[:, 1:], pe)
                input_set = input_set.reshape(1, self.backword_days, self.number_of_features)
                day += 1
            
            predictions = np.array(predictions).reshape(days, self.number_of_features)
            self.one_by_one_df = pd.DataFrame(predictions, columns=self.significant_features,
                                              index=self.get_dates(begin_date, days))
            self.one_by_one_df.reset_index(inplace=True)
            self.one_by_one_df.rename(columns={"index":"Date"}, inplace=True)

            return self.one_by_one_df
    
    def change_dataset(self, new_dataset: pd.DataFrame) -> None:
        self.raw_dataset = new_dataset

    def load_model(self, folder_name: str) -> bool:
        """
            Description: 

            Parameters
            ----------

            
            Returns
            -------
            
        """
        cwd = os.getcwd().replace("\\", "/")
        folder_path = cwd + "/StockPredictorLSTM/DATA/"+folder_name
        print(folder_path)
        if not os.path.exists(folder_path):
            print("No data to load")
            return False
        else:
            metrics = None
            with open(folder_path+"/metrics.p", "rb") as handler:
                metrics = pickle.load(handler)
            
            self.error_distribution = metrics["error_dist"]
            self.scaler = metrics["scaler"]
            self.significant_features = metrics["features"]
            self.backword_days = metrics["backword_days"]
            self.number_of_features = metrics["features_number"]
            self.RMSE = metrics["rmse"]
            self.raw_dataset = metrics["raw_dataset"]
            del metrics
            self.model = load_model(folder_path+"/model.h5")
            print("Model summary:\n", self.model.summary())
            
            return True

    def save_model(self, folder_name: str) -> bool:
        """
            Description: 

            Parameters
            ----------

            
            Returns
            -------
            
        """
        if self.model:
            metrics = {
                "error_dist": self.error_distribution,
                "scaler": self.scaler,
                "features": self.significant_features,
                "backword_days": self.backword_days,
                "features_number": self.number_of_features,
                "rmse": self.RMSE,
                "raw_dataset": self.raw_dataset,
            }
            cwd = os.getcwd().replace("\\", "/")
            folder_path = cwd + "/StockPredictorLSTM/DATA/" + folder_name
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            with open(folder_path+"/metrics.p", "wb") as handler:
                pickle.dump(metrics, handler)
            self.model.save(folder_path+"/model.h5")

            return True
        else:
            print("No model to save.")
            return False

    def initialize_model(self, shape: tuple) -> None:
        """
            Description: 

            Parameters
            ----------

        """
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', return_sequences=True, bias_regularizer=regularizers.l2(1e-4), 
                        activity_regularizer=regularizers.l2(1e-5), input_shape=shape))
        self.model.add(Dropout(0.15))
        self.model.add(LSTM(50, activation='relu', return_sequences=True, bias_regularizer=regularizers.l2(1e-4), 
                        activity_regularizer=regularizers.l2(1e-5)))
        self.model.add(Dropout(0.1))
        self.model.add(LSTM(50, activation='relu', return_sequences=True, bias_regularizer=regularizers.l2(1e-4), 
                        activity_regularizer=regularizers.l2(1e-5)))
        self.model.add(Dropout(0.05))
        self.model.add(LSTM(50, activation='relu', bias_regularizer=regularizers.l2(1e-4), 
                        activity_regularizer=regularizers.l2(1e-5)))
        self.model.add(Dropout(0.05))
        self.model.add(Dense(shape[1]))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.summary()

    def get_xy_sets(self, data_set: np.array, batch_size: int) -> Tuple[np.array, np.array]:
        """
            Description: 

            Parameters
            ----------

            
            Returns
            -------
            
        """
        x = []  # dependent
        y = []  # independent
        dataset_size = len(data_set)
        try:
            if dataset_size < self.backword_days: raise Exception("Dataset too small")
            for i in range(batch_size, dataset_size):
                x.append(data_set[i-batch_size:i])
                y.append(data_set[i])
            return np.array(x), np.array(y)
        except Exception("Dataset too small"):
            print("Your dataset size: {}\nMinimum dataset size reqired: {}".format(dataset_size, self.backword_days))
            return None, None
    
    def get_dates(self, beginning: str, days_forword: int) -> list:
        """
            Description: 

            Parameters
            ----------

            
            Returns
            -------
            
        """
        dates = []
        day = datetime.datetime.strptime(beginning, "%Y-%m-%d").date()
        holis = list(holidays.US(years=datetime.datetime.now().year).keys())
        while len(dates) < days_forword:
            if day not in holis and day.weekday() < 5:
                dates.append(day)
            day += datetime.timedelta(days=1)
        return dates

    def display_info(self, error_boxplot: bool=False) -> None:
        """
            Description: 

            Parameters
            ----------

            
            Returns
            -------
            
        """
        print("\n\tINFO:")
        if self.training_time is not None: print("\nTraining time: {:.5f}s".format(self.training_time))
        print("\nRMSE for each feature:\n", self.RMSE)
        print("Lowest RMSE feature: {}".format(self.RMSE[['RMSE [%]']].idxmin()[0]))
        print("\nError distribution:\n",
                    pd.DataFrame(self.error_distribution, columns=self.significant_features).describe())
        
        if error_boxplot:
            plt.title("\nError distribution")
            plt.boxplot(self.error_distribution, labels=self.significant_features)
            plt.show()      

    def prediction_plot(self, feature: str, COMPANY_NAME: str, forword_days: int) -> None:
        """
            Description: 

            Parameters
            ----------

            
            Returns
            -------
            
        """
        if self.one_by_one_df is not None:
            to_plot = self.one_by_one_df.set_index("Date")
            plt.figure(figsize=(16,8))
            ax = sb.lineplot(data=to_plot[[feature]], marker="o")
            ax.set_xticklabels(to_plot.index, rotation=45)
            ax.set(xticks=to_plot.index)
            ax.xaxis.set_major_formatter(dates.DateFormatter("%d-%m"))
            plt.legend(["Predicted close prices"])
            plt.xlabel("Date")
            plt.ylabel("Price [$]")
            plt.title("{}: {} for next {} days".format(COMPANY_NAME, feature, forword_days))

            for x, y in zip(to_plot.index, to_plot[[feature]].values):
                label = "{:.2f}".format(y[0])
                plt.annotate(label, (x,y), textcoords="offset points", xytext=(0,10), ha='center')

            plt.show()
        else:
            print("No data to plot.")
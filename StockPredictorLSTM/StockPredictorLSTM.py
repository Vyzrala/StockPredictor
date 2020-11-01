import os
import time
import random
import pickle
import holidays
import datetime
import numpy as np
import pandas as pd
import seaborn as sb
from scipy import stats
from typing import Tuple
from matplotlib import dates
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from keras import regularizers
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from sklearn.utils import validation
from tensorflow.python.keras.optimizers import Optimizer

class Predictor:
    """
        This is a thesis projects for bechelor degree at Gdańsk University of technology in Poland.
        The Predictor class provides all nessecery methods and functionality for predicting future values of stocks for given company.
        All you need to do is insert a data set and excute predict() method.
        The creators of Predictor class are: Patryk Dunajewski, Marcin Hebdzyński and Kamil Rutkowski.
    """

    def __init__(self, correlation_threshold: float=0.75, split_ratio: float=0.8, backword_days: int=60,
                 epochs_number: int=120, batch: int=32, error_impact: float=0.8) -> None:
        """
            Description: Initialization method where you can specify many parameters

            Parameters
            ----------
                correlaction_threshold : float
                    threshold value <0, 1> that filter out features model will be trained on
                split_ratio : float
                    value <0, 1> that split dataset in test and test sets with specified ratio
                backword_days : int
                    number of days that model will require to predict further values
                epochs_number : int
                    number of epochs that model will be trained on
                batch : int
                    number of samples per batch of computation
                error_impact : float
                    how much of error value will be added to predicted value (noice)
        """
        self.raw_dataset = None
        self.correlation_threshold = correlation_threshold
        self.train_data = None
        self.test_data = None
        self.split_ratio = split_ratio
        self.backword_days = backword_days
        self.epochs_number = epochs_number
        self.first_training_time = 0
        self.final_training_time = 0
        self.total_training_time = 0
        self.batch = batch
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.rmse = None
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
                raw dataset that model will use to train itself
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
        print("First training:")
        start_time = time.time()
        self.model.fit(x_train, y_train, epochs=self.epochs_number, batch_size=self.batch, validation_split=0.08)
        self.first_training_time = time.time() - start_time
        print("First training time: {:.2f} minutes ({:.3f}s)".format(self.first_training_time/60, self.first_training_time))

        # Testing model on test set
        x_test, y_test = self.get_xy_sets(self.test_data, self.backword_days)
        y_predictions = None
        if x_test is None and y_test is None:
            return
        y_predictions = self.model.predict(x_test)

        # Model evaluation
        y_predictions = self.scaler.inverse_transform(y_predictions)
        y_test = self.scaler.inverse_transform(y_test)
        self.rmse = pd.DataFrame([np.sqrt(np.mean((y_test[:,i] - y_predictions[:,i])**2)) for i in range(y_test.shape[1])],
                                 index=self.significant_features, columns=['RMSE [%]'])
        print("RMSE:")
        print(self.rmse)

        # Error distribution
        self.error_distribution = y_test - y_predictions
        self.error_distribution = self.error_distribution[(np.abs(stats.zscore(self.error_distribution))<3).all(axis=1)]
        
        # Final training
        final_dataset = self.scaler.fit_transform(dataset)
        final_x, final_y = self.get_xy_sets(final_dataset, self.backword_days)
        print("\nFinal training:")
        start_time = time.time()
        self.model.fit(final_x, final_y, epochs=self.epochs_number, batch_size=self.batch, validation_split=0.1)
        self.final_training_time = time.time() - start_time
        print("Final traning time: {:.2f} minutes ({:.3f}s)".format(self.final_training_time/60, self.final_training_time))
        self.total_training_time = self.final_training_time + self.first_training_time

    def predict(self, days: int) -> pd.DataFrame: 
        """
            Description: Method predicts future values 

            Parameters
            ----------
                days : int
                    number of days that model will predict further
            
            Returns
            -------
                dataset with predicted values

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
                p = self.model.predict(input_set)  # Predict future value
                p = self.scaler.inverse_transform(p)  # Unscale predicted value
                predictions.append(p)  # Save predicted and unscaled value to temporary variable
                p += random.choice(self.error_distribution * self.error_impact)  # Add random error value to predicted value
                pe = self.scaler.transform(p)  # Transform preidcted value with error to range <0, 1>
                input_set = np.append(input_set[:, 1:], pe)  # Modify dataset, add predicted value to dataset
                input_set = input_set.reshape(1, self.backword_days, self.number_of_features)  # Update shape of dataset that model will preodict from
                day += 1  # Increment iterator
            
            predictions = np.array(predictions).reshape(days, self.number_of_features)
            self.one_by_one_df = pd.DataFrame(predictions, columns=self.significant_features,
                                              index=self.get_dates(begin_date, days))
            self.one_by_one_df.reset_index(inplace=True)
            self.one_by_one_df.rename(columns={"index":"Date"}, inplace=True)
            self.one_by_one_df.Date = pd.to_datetime(self.one_by_one_df.Date)
            return self.one_by_one_df
    
    def load_model(self, folder_name: str) -> bool:
        """
            Description: Loads data about specific model

            Parameters
            ----------
                folder_name : str
                    name of folder where data about model will be saveds
            
            Returns
            ------- 
                boolean value according to succes of failure of loading data
        """
        cwd = os.getcwd().replace("\\", "/")
        folder_path = cwd + "/StockPredictorLSTM/DATA/"+folder_name
        if not os.path.exists(folder_path):
            print("No data to load")
            return False
        else:
            metrics = {}
            with open(folder_path+"/metrics.p", "rb") as handler:
                metrics = pickle.load(handler)
            
            self.error_distribution = metrics.get('error_dist', None)
            self.scaler = metrics.get('scaler', None)
            self.significant_features = metrics.get('features', None)
            self.backword_days = metrics.get('backword_days')
            self.number_of_features = metrics.get('features_number', None)
            self.rmse = metrics.get('rmse', None)
            self.raw_dataset = metrics.get('raw_dataset', None)
            self.total_training_time = metrics.get('total_training_time', None)
            del metrics
            self.model = load_model(folder_path+"/model.h5")
            self.model.compile(optimizer='adam', loss='mean_squared_error')
            print("Model summary:\n", self.model.summary())
            
            return True

    def save_model(self, folder_name: str) -> bool:
        """
            Description: Save data about actually trained model

            Parameters
            ----------
                folder_name : str
                    name of dictionary where data about model will be saved
            
            Returns
            -------
                boolean value according to success of failure of action
        """
        if self.model:
            metrics = {
                "error_dist": self.error_distribution,
                "scaler": self.scaler,
                "features": self.significant_features,
                "backword_days": self.backword_days,
                "features_number": self.number_of_features,
                "rmse": self.rmse,
                "raw_dataset": self.raw_dataset,
                'total_training_time': self.total_training_time
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

    def initialize_model(self, shape: Tuple[int, int]) -> None:
        """
            Description: Method initialize structure of model

            Parameters
            ----------
                shape : tuple of integers
                    shape of training dataset
        """
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', return_sequences=True, 
                        bias_regularizer=regularizers.l2(1e-4), 
                        activity_regularizer=regularizers.l2(1e-5), input_shape=shape))
        self.model.add(Dropout(0.15))
        self.model.add(LSTM(50, activation='relu', return_sequences=True, 
                        bias_regularizer=regularizers.l2(1e-4), 
                        activity_regularizer=regularizers.l2(1e-5)))
        self.model.add(Dropout(0.1))
        self.model.add(LSTM(50, activation='relu', return_sequences=True, 
                        bias_regularizer=regularizers.l2(1e-4), 
                        activity_regularizer=regularizers.l2(1e-5)))
        self.model.add(Dropout(0.05))
        self.model.add(LSTM(50, activation='relu', 
                        bias_regularizer=regularizers.l2(1e-4), 
                        activity_regularizer=regularizers.l2(1e-5)))
        self.model.add(Dropout(0.05))
        self.model.add(Dense(shape[1]))

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.summary()

    def change_dataset(self, new_dataset: pd.DataFrame) -> None:
        """
            Descruption: Method changes operating dataset for new inserted one, if the columns in both are same

            Parameters
            ----------
                new_dataset : pandas DataFrame
                    new dataset that will be replaced with old one if it is possible
        """

        if all(col in self.raw_dataset for col in new_dataset.columns):
            self.raw_dataset = new_dataset
            return True
        else:
            print("Fail to change dataset")
            return False
    
    def download_dataset(self, beginning_date: str, end_date: str, company: str) -> pd.DataFrame:
        """
            Description: Method downloads data do pandas DataFrame from https://finance.yahoo.com/. 
                         
            Parameters
            ----------
                beginning_date : str
                    Date from which data will be downloaded
                end_date : str
                    Date till which data will be downloaded
                company : str
                    company name of whose data will be downloaded
            
            Returns
            -------
                pandas DataFrame object with requested data in form of 
                    index: Timestamp
                    columns: High, Low, Open, Close, Volume, Adj Close
        """

        source = 'yahoo'
        dataset = pdr.DataReader(company, source, beginning_date, end_date)
        dataset.reset_index(inplace=True)
        return dataset

    def get_xy_sets(self, dataset: np.array, batch_size: int) -> Tuple[np.array, np.array]:
        """
            Description: Method splits test and train data into two sets of dependent and independent variables

            Parameters
            ----------
                dataset : numpy array
                    dataset in form of numpy array
                batch_size : int
                    number of samples in single batch for training
            Returns
            -------
                Two numpy arrays of splited dataset into x and y sets
        """
        x = []  # dependent
        y = []  # independent
        dataset_size = len(dataset)
        try:
            if dataset_size < self.backword_days: raise Exception("Dataset too small")
            for i in range(batch_size, dataset_size):
                x.append(dataset[i-batch_size:i])
                y.append(dataset[i])
            return np.array(x), np.array(y)
        except Exception("Dataset too small"):
            print("Your dataset size: {}\nMinimum dataset size reqired: {}".format(dataset_size, self.backword_days))
            return None, None
    
    def get_dates(self, beginning: str, days_forword: int) -> list:
        """
            Description: Generates list of dates for given number of days from beginning date ommiting holidays and weekends

            Parameters
            ----------
                beginning : str
                    date from with dates will be generated
                days_forword : int
                    number of working days futher form beginning date
            Returns
            -------
                List of dates
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
            Description: Method displays information about model such as: training time, RMSE for each feature, error distribution

            Parameters
            ----------
                error_boxplot : bool
                    parameter that decide wether to display or not error distribution boxplots for each feature
        """
        print("\n\tINFO:\n")
        if self.first_training_time is not None: print("First training time: {:.5f}s".format(self.first_training_time))
        if self.final_training_time is not None: print("Final training time: {:.5f}s".format(self.final_training_time))
        if self.total_training_time is not None:
            print("Total training time: {:.2f} minutes ({:.3f}s)".format(self.total_training_time/60, self.total_training_time))

        print("\nRMSE for each feature:\n", self.rmse)
        print("Lowest RMSE feature: {}".format(self.rmse[['RMSE [%]']].idxmin()[0]))
        print("\nError distribution:\n",
                    pd.DataFrame(self.error_distribution, columns=self.significant_features).describe())
        
        if error_boxplot:
            plt.title("\nError distribution")
            plt.boxplot(self.error_distribution, labels=self.significant_features)
            plt.show()      

    def prediction_plot(self, feature: str, company_name: str, forword_days: int) -> None:
        """
            Description: Method displays plot of predicted values if such data exists

            Parameters
            ----------
                feature : str 
                    feature that we want to visualize
                company_name : str
                    name of company to which the data relates
                forword_days : int
                    number of days to display
        """
        if self.one_by_one_df is not None and \
            feature in self.one_by_one_df.columns and \
            forword_days == self.one_by_one_df.shape[0]:

            to_plot = self.one_by_one_df.set_index("Date")
            fig = plt.figure(figsize=(16,8))
            fig.canvas.set_window_title("{} predictions for next {} days".format(company_name, forword_days))
            ax = sb.lineplot(data=to_plot[[feature]], marker="o")
            ax.set_xticklabels(to_plot.index, rotation=20)
            ax.set(xticks=to_plot.index)
            ax.xaxis.set_major_formatter(dates.DateFormatter("%d-%m"))
            plt.legend(["Predicted close prices"])
            plt.xlabel("Date [dd-mm]")
            plt.ylabel("Price [$]")
            plt.title("{}: {} for next {} days".format(company_name, feature, forword_days))

            for x, y in zip(to_plot.index, to_plot[[feature]].values):
                label = "{:.2f}".format(y[0])
                plt.annotate(label, (x,y), textcoords="offset points", xytext=(0,10), ha='center')

            plt.show()
        else:
            print("\nERROR\n----------------------------------------")
            print("Your feature: {} | Availabe features: {}".format(feature, list(self.one_by_one_df.columns)))
            print("Your days forword: {} | Available days forword: {}\n".format(forword_days, self.one_by_one_df.shape[0]))

    def compare_directions(self, predictions, valid_set, feature) -> dict:
        """
            Description: This method perform simulation of correctly predicted direction of prices between days.
                         You need a set of valid data that could be compared with predictions. 
                         This is an accuracy measure function of out project.
            
            Parameters
            ----------
                predictions : pandas DataFrame
                    A dataset containing predicted values
                valid_set : pandas DataFrame
                    A dataset containing valid values. Will be compared with predictions
                feature : str
                    Feature that will be tested
            
            Returns
            -------
                Dictionary with metrics describing correctness of predicted ups and downs
        """
        def graph_directions(dataset):
            directions = []
            for i in range(len(dataset)-1):
                direction = dataset[i+1:i+2] - dataset[i:i+1]
                if direction > 0: directions.append(1)  # UP
                elif direction == 0: directions.append(0)  # CONST
                else: directions.append(-1)  # DOWN
            return np.array(directions)

        if len(predictions[feature]) != len(valid_set[feature]): 
            print("Wrong input")
            return
        predictions = graph_directions(predictions[feature].values)
        valid_set = graph_directions(valid_set[feature].values)
        comparison = list(map(lambda x: 1 if x else 0, predictions == valid_set))
        correct = sum(comparison)
        cases = len(comparison)
        rv = {"Correctness [%]": round(100*correct/cases, 3),
              "Correct": correct,
              "Cases": cases,
              "Distribution": comparison}
        return rv
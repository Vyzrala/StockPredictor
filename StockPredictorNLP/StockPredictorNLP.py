import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM, GRU, SimpleRNN
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import time, datetime, os
from typing import Tuple


class PredictorNLP:
    def __init__(self, 
                 split_ratio: float=0.8, 
                 epochs_number: int=10) -> None:
        self.layers = [LSTM, GRU, SimpleRNN]
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.features = ['Open', 'Close', 'Volume', 'Polarity', 'Subjectivity']
        self.split_ratio = split_ratio
        self.epochs_number = epochs_number
        self.raw_dataset = None
        self.model = None
        self.models = {}
        self.best_model_stuff = {}
        self.rmse = 0
        self.last_prediction = 0
    
    def create_model(self, dataset: pd.DataFrame):
        self.raw_dataset = dataset.copy()
        
        # Process dataset
        dfa = np.array(dataset[self.features].copy())
        split_index = int(len(dfa) * self.split_ratio)
        train_set = dfa[:split_index, :].copy()
        test_set = dfa[split_index:, :].copy()
        self.scaler.fit(train_set)
        train_scaled = self.scaler.transform(train_set)
        test_scaled = self.scaler.transform(test_set)
        x_train, y_train = self.get_xy_sets(train_scaled)
        x_test, y_test = self.get_xy_sets(test_scaled)
        print(x_train.shape)
        shape = (x_train.shape[1], x_train.shape[2])
        
        # create models
        models = {}
        for layer in self.layers:
            tmp_model = self.initialize_model(layer, shape)
            tmp_history = tmp_model.fit(x_train, y_train, 
                                epochs=self.epochs_number, 
                                batch_size=1, 
                                validation_split=.05)
            tmp_predictions = tmp_model.predict(x_test)
            unscaled_predictions = ((tmp_predictions.flatten() - self.scaler.min_[0]) 
                                    / self.scaler.scale_[0])
            unscaled_y_test = ((y_test.flatten() - self.scaler.min_[0]) 
                               / self.scaler.scale_[0])
            tmp_rmse = np.sqrt(mean_squared_error(unscaled_y_test, unscaled_predictions))
            print("\n{}: RMSE = {}\n".format(str(layer).split('.')[-1][:-2], tmp_rmse))
            models[layer] = {'model': tmp_model, 
                             'history': tmp_history, 
                             'predictions': unscaled_predictions,
                             'rmse':tmp_rmse,
                             'layer':str(layer).split('.')[-1][:-2]}

        # select best model
        self.models = models
        best_model_stuff = min(models.values(), key=lambda x: x['rmse'])
        print(best_model_stuff)
        
        # Final training
    
    def predict(self):
        pass
    
    def save_model(self, path: str):
        pass
    
    def load_model(self, path: str):
        pass

    def get_data_from_file(self, path_to_file):
        dataset = None
        if os.path.isfile(path_to_file):
            dataset = pd.read_csv(path_to_file)  
            return dataset          
        else:
            raise NLPError('\nNo such file.\nPlease check if file at {} exists.'.format(path_to_file))
    
    def get_xy_sets(self, dataset: np.array) -> Tuple[np.array, np.array]:
        x, y = [], []
        for i in range(len(dataset)-1):
            x.append(np.array([dataset[i, 1:]]))
            y.append(np.array([dataset[i+1, 0]]))
        
        return (np.array(x), np.array(y))
    
    def initialize_model(self, layer, shape, show_summary=False) -> None:
        tf.keras.backend.clear_session()
        model = Sequential()
        model.add(layer(50, activation='relu', return_sequences=True, input_shape=shape))
        model.add(layer(50, activation='relu'))
        model.add(Dense(1))
        opt = tf.keras.optimizers.Adam()
        model.compile(optimizer=opt, 
                      loss='mean_squared_error', 
                      metrics=['accuracy', 
                               tf.keras.metrics.RootMeanSquaredError()])
        if show_summary: 
            model.summary()
        return model


class NLPError(Exception): pass
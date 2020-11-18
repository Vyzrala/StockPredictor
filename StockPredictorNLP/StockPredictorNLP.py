from keras.layers import Dense, LSTM, GRU, SimpleRNN
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import tensorflow as tf
import seaborn as sb
import pandas as pd
import datetime, os
import numpy as np

class PredictorNLP:
    def __init__(self, epochs_number: int=10,
                 split_ratio: float=0.8, 
                 batch_size: int=1) -> None:
        self.layers = [LSTM, GRU, SimpleRNN]
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.features = ['Open', 'Close', 'Volume', 'Polarity', 'Subjectivity']
        self.split_ratio = split_ratio
        self.epochs_number = epochs_number
        self.batch_size = batch_size
        self.raw_dataset = pd.DataFrame()
        self.prediction = pd.DataFrame()
        self.models = {}
        self.best_model_data = {}
        self.model = None
        self.rmse = 0
        self.company_name = None
        self.y_test = None
    
    def create_model(self, dataset: pd.DataFrame) -> None:
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
        
        shape = (x_train.shape[1], x_train.shape[2])
        unscale = lambda x: ((x.flatten() - self.scaler.min_[0]) 
                             / self.scaler.scale_[0])
        self.y_test = unscale(y_test)
        # create models
        models = {}
        for layer in self.layers:
            layer_name = str(layer).split('.')[-1][:-2]
            tmp_model = self.initialize_model(layer, shape)
            tmp_history = tmp_model.fit(x_train, y_train, 
                                        epochs=self.epochs_number, 
                                        batch_size=self.batch_size, 
                                        validation_split=.05)

            tmp_predictions = tmp_model.predict(x_test)
            unscaled_predictions = unscale(tmp_predictions)
            unscaled_y_test = unscale(y_test)
            
            tmp_rmse = np.sqrt(mean_squared_error(unscaled_y_test, unscaled_predictions))
            msg = '{}: RMSE = {}'.format(layer_name, tmp_rmse)
            print('\n\t', msg, '\n')
            models[layer_name] = {'layer': layer_name,
                                  'model': tmp_model, 
                                  'rmse': tmp_rmse,
                                  'history': tmp_history, 
                                  'predictions': unscaled_predictions}

        # Select best model by the lowest RMSE value
        best_model_stuff = min(models.values(), key=lambda x: x['rmse'])
        self.models = models
        
        # Final training
        final_dataset = self.scaler.fit_transform(dfa)
        final_x, final_y = self.get_xy_sets(final_dataset)
        model = best_model_stuff['model']
        model.fit(final_x, final_y, 
                  epochs=self.epochs_number,
                  batch_size=self.batch_size,
                  validation_split=.05)
        
        model.summary()
        self.model = model
        self.rmse = best_model_stuff['rmse']
        self.best_model_data = best_model_stuff
        print('\nBest model ({}) has been fully trained.'.format(best_model_stuff['layer']))
    
    def predict(self) -> pd.DataFrame:
        if not self.model:
            raise NLPError("Model has not been initialized.")
        if ((not self.raw_dataset.empty) and 
            (not isinstance(self.raw_dataset, pd.DataFrame))):
            raise NLPError('No dataset or incorrect dataset type')
        try:
            last_row = self.raw_dataset.iloc[-1]
            
            input_values = np.array(last_row[self.features])
            input_values = input_values.reshape(1, len(self.features))  # (1, 5)
            input_values = self.scaler.transform(input_values)
            input_values = input_values[:, 1:].reshape(1, 1, len(self.features)-1)  # (1, 1, 4)
                
            pred = self.model.predict(input_values)
            pred = ((pred.flatten() - self.scaler.min_[0]) / self.scaler.scale_[0])[0]

            next_day = pd.to_datetime(last_row['Date']) + datetime.timedelta(days=1)
            while next_day.weekday() > 4:
                next_day += datetime.timedelta(days=1)
            
            self.prediction = pd.DataFrame([[next_day, pred]], columns=['Date', 'Open'])
            self.prediction.Date = pd.to_datetime(self.prediction.Date)
            return self.prediction
        except:
            raise NLPError('Error while predicting.')
    
    def save_model(self, path: str) -> bool:
        pass
    
    def load_model(self, path: str) -> bool:
        pass

    def get_data_from_file(self, path_to_file: str) -> pd.DataFrame:
        path_to_file = path_to_file.replace('\\', '/')
        path_to_file = path_to_file.replace('//', '/')
        if os.path.isfile(path_to_file):
            dataset = pd.read_csv(path_to_file) 
            if all(feat in dataset.columns for feat in self.features):
                self.company_name = path_to_file.split('/')[-1].split('.')[0]
                return dataset          
            else:
                raise NLPError('Wrong dataset.')
        else:
            raise NLPError('\nNo such file.\nPlease check if file exists at {}.'.format(path_to_file))
    
    def get_xy_sets(self, dataset: np.array) -> Tuple[np.array, np.array]:
        x, y = [], []
        for i in range(len(dataset)-1):
            x.append(np.array([dataset[i, 1:]]))
            y.append(np.array([dataset[i+1, 0]]))
        
        return (np.array(x), np.array(y))
    
    def initialize_model(self, layer, shape):
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
        return model
    
    def get_models_comparison_graph(self):
        if not (self.models or self.best_model_data): 
            raise NLPError('No models data tu display.')
        
        preds, names = [], []
        for model_data in self.models.values():
            preds.append(model_data['predictions'])
            names.append(model_data['layer'])
        
        legend = ['Actual values'] + names
        best_model_ = self.best_model_data['layer']
        best_model_index = legend.index(best_model_)
        legend[best_model_index] += ' (BEST)'
        
        plt.figure(figsize=(15, 8))            
        sb.lineplot(data=self.y_test, linewidth=4,marker='o', color='blueviolet')  # valid
        sb.lineplot(data=preds, marker='o')        
        sb.color_palette('rocket')
        plt.legend(legend)
        plt.title(f'Models comparison on test data for {self.company_name}')
        plt.xlabel('Days')
        plt.ylabel('Open price [$]')
        plt.show()

class NLPError(Exception): pass
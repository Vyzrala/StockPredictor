from StockPredictorLSTM.StockPredictorLSTM import LSTMError
import unittest
import pandas_datareader as pdr
import datetime
import os
import pandas as pd

from StockPredictorLSTM import PredictorLSTM

## RUN: python -m unittest

class X_TestStockPredictorLSTM(unittest.TestCase):
    settings = {
        'company_name': 'FB',
        'start': '2017-01-01',
        'end': '2018-01-01',
        'days': 15,
    }
    model = PredictorLSTM(epochs_number=3)
    test_data = model.download_dataset(settings['start'], 
                                       settings['end'], 
                                       settings['company_name'])
    model.create_model(test_data)
    
    def test_predict(self):
        predictions  = self.model.predict(self.settings['days'])
        predictions_size = len(predictions)
        self.assertIsInstance(predictions, pd.DataFrame, msg='Data type error')
        # self.assertEqual(predictions_size, self.settings['days'], msg='Predictions not equal predicted days')
    
    def X_test_predict_raise(self):
        local_model = PredictorLSTM()
        self.assertRaises(LSTMError('No dataset'), local_model.predict, self.settings['days'])
    
    def test_save_model(self):
        
        file_dir = "test/"+self.settings['company_name']
        self.model.save_model(file_dir)
        
        cwd = os.getcwd().replace("\\", "/")
        metrics_path = cwd + "/StockPredictorLSTM/DATA/" + file_dir +"/metrics.p"
        model_path = cwd + "/StockPredictorLSTM/DATA/" + file_dir +"/model.h5"
        metrics_bool = os.path.isfile(metrics_path)
        model_bool = os.path.isfile(model_path)

        self.assertTrue(metrics_bool, "No metrics file")
        self.assertTrue(model_bool, "No model file")
        
        # Clean up after yourself
        try:
            os.remove(metrics_path)
            os.remove(model_path)
            os.rmdir(model_path[:-9])
            os.rmdir(model_path[:-12])
        except:
            print("Error when removing")

    
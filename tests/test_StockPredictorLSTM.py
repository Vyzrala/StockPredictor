import unittest
import pandas_datareader as pdr
import datetime
import os
from StockPredictorLSTM import Predictor

## RUN: python -m unittest

class TestStockPredictorLSTM(unittest.TestCase):
    def test_predict(self):
        COMPANY_NAME = 'FB'
        START_DATE = '2017-01-01'
        END_DATE = str(datetime.datetime.today().date())
        SOURCE = 'yahoo'
        DAYS = 15
        test_data = pdr.DataReader(COMPANY_NAME, SOURCE, START_DATE, END_DATE)
        test_data.reset_index(inplace=True)
        
        model = Predictor(epochs_number=5)
        model.create_model(test_data)
        model.display_info()
        print("\n{} days forword:\n".format(DAYS), model.predict(DAYS))
        model.prediction_plot("Close", COMPANY_NAME, DAYS)
    
    def test_save_model(self):
        COMPANY_NAME = 'FB'
        START_DATE = '2017-01-01'
        END_DATE = '2018-05-04' #str(datetime.datetime.today().date())
        SOURCE = 'yahoo'
        test_data = pdr.DataReader(COMPANY_NAME, SOURCE, START_DATE, END_DATE)
        test_data.reset_index(inplace=True)
        model = Predictor(epochs_number=5)
        model.create_model(test_data)
        file_dir = "test/"+COMPANY_NAME
        model.save_model(file_dir)
        
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

    def test_dataset_size(self):
        COMPANY_NAME = 'FB'
        START_DATE = '2017-01-01'
        END_DATE = str(datetime.datetime.today().date())
        SOURCE = 'yahoo'
        DAYS = 15
        test_data = pdr.DataReader(COMPANY_NAME, SOURCE, START_DATE, END_DATE)[-10:]
        test_data.reset_index(inplace=True)
        model = Predictor(epochs_number=5)
        self.assertRaises(Exception, model.create_model, test_data)

    def test_load_model(self):
        pass
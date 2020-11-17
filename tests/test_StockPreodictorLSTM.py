import unittest
import pandas_datareader as pdr
import datetime
import os
from StockPredictorLSTM import PredictorLSTM

## RUN: python -m unittest

class TestStockPredictorLSTM(unittest.TestCase):
    
    def setUp(self) -> None:
        self.settings = {
            'company_name': 'FB',
            'start': '2017-01-01',
            'end': str(datetime.datetime.today().date()),
            'source': 'yahoo',
            'days': 15,
        }
        self.test_data = pdr.DataReader(self.settings['company_name'], 
                                        self.settings['source'], 
                                        self.settings['start'], 
                                        self.settings['end'])
        self.test_data.reset_index(inplace=True)
        self.model = PredictorLSTM(epochs_number=5)
        self.model.create_model(self.test_data)
        self.model.display_info()
    
    def test_predict(self):
        print("\n{} days forword:\n".format(self.settings['days']), self.model.predict(self.settings['days']))
        
    
    def test_save_model(self):
        COMPANY_NAME = 'FB'
        START_DATE = '2017-01-01'
        END_DATE = '2018-05-04' #str(datetime.datetime.today().date())
        SOURCE = 'yahoo'
        test_data = pdr.DataReader(COMPANY_NAME, SOURCE, START_DATE, END_DATE)
        test_data.reset_index(inplace=True)
        model = PredictorLSTM(epochs_number=5)
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
        test_data = pdr.DataReader(COMPANY_NAME, SOURCE, START_DATE, END_DATE)[-10:]
        test_data.reset_index(inplace=True)
        model = PredictorLSTM(epochs_number=5)
        self.assertRaises(Exception, model.create_model, test_data)

    def test_load_model(self):
        pass
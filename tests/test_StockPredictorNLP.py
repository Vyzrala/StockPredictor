from StockPredictorNLP.StockPredictorNLP import NLPError
import unittest
import pandas as pd
from StockPredictorNLP import *


class TestStockPredictorNLP(unittest.TestCase):
    predictor = PredictorNLP()
    
    def x_test_get_data_from_file(self):
        file_path = '/XD'
        self.assertRaises(NLPError, lambda: self.predictor.get_data_from_file(file_path))  # Rise NLPError
        file_path = '/home/marcin/Documents/python_projects/StockPredictor/\
            StockPredictor/tests/test_data/2020.11.15_BFFMT.csv'
        result = self.predictor.get_data_from_file(file_path)
        self.assertIsInstance(result, pd.DataFrame)  # Check if result is corrent type
        
    def test_similation(self):
        path = '/home/marcin/Documents/python_projects/StockTest/data/AAPL.csv'
        predictor = PredictorNLP()
        
        dataset = predictor.get_data_from_file(path)
        predictor.create_model(dataset)
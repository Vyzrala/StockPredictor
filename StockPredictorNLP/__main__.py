#!/usr/bin/env python3
from StockPredictorNLP import PredictorNLP
from StockPredictorNLP.tools import preprocess_raw_datasets


def main():
    path = 'D:\Programing projects\Thesis datasets\Pre-processed data\AAPL.csv'
    predictor = PredictorNLP()        
    dataset = predictor.get_data_from_file(path)
    predictor.create_model(dataset)
    prediction = predictor.predict()   
    print(prediction)
    

if __name__ == '__main__':
    main()
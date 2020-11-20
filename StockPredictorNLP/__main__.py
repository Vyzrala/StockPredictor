#!/usr/bin/env python3
from StockPredictorNLP import PredictorNLP
from .tools import preprocess_raw_datasets


def main():
    path = 'D:\Programing projects\Thesis datasets\Pre-processed data\AAPL.csv'
    # preprocess_raw_datasets('D:\Programing projects\Thesis datasets', 'Pre-processed data')
    
    predictor = PredictorNLP()        
    dataset = predictor.get_data_from_file(path)
    predictor.create_model(dataset)
    prediction = predictor.predict()   
    print(prediction)
    predictor.save_model('/test')
    
    # predictor.load_model('/test')
    # print(predictor.predict())

if __name__ == '__main__':
    main()
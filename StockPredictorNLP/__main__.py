#!/usr/bin/env python3
from StockPredictorNLP import PredictorNLP


def main():
    path = '/home/marcin/Documents/python_projects/StockTest/data/AAPL.csv'
    predictor = PredictorNLP()
    
    dataset = predictor.get_data_from_file(path)
    predictor.create_model(dataset)


if __name__ == '__main__':
    main()
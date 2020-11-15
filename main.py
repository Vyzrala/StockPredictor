#!/usr/bin/env python3

# from StockPredictorLSTM import PredictorLSTM
# from StockPredictorNLP import PredictorNLP
from StockPredictorNLP.tools import preprocess_raw_datasets

def main():
    path = '/media/marcin/DANE/Programing projects/Thesis datasets'
    preprocess_raw_datasets(path, '')


if __name__ == '__main__':
    main()

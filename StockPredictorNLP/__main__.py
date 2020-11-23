#!/usr/bin/python3 

from StockPredictorNLP.tools import preprocess_raw_datasets, read_files_and_yahoo
import time

path_to_raw = '/media/marcin/DANE/Programing projects/Thesis datasets'

t0 = time.time()
all_tweets, yahoo_data = read_files_and_yahoo(path_to_raw)
sentiment_data = preprocess_raw_datasets(all_tweets, yahoo_data)
del all_tweets, yahoo_data
print('\n Processing time: {:.3f}s'.format(time.time() - t0))
for k, v in sentiment_data.items():
    print(' -> {}: \t{}'.format(k, v.shape))
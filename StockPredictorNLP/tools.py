from os import curdir
from typing import List
import pandas as pd
import numpy as np
import glob, sys, os, logging, datetime, re, copy
import pandas_datareader as pdr
from textblob import TextBlob
from pathlib import Path

settings = {
		'format': '%(acstime)s | %(levelname)s | %(message)s',
		'log_file': os.getcwd() + '/NLP/tools.log',
}
# Path(settings['log_file']).mkdir(parents=True, exist_ok=True)
# logging.basicConfig(filename=settings['log_file'],
# 					level=logging.INFO,
# 					format=settings['format'])

companies_keywords = {
	'AAPL': [' Apple ','Iphone','MacOS','Ipod','Ipad','AirPods','HomePod',
			 'Arthur Levinson','Tim Cook', 'Steve Jobs','Steve Wozniak',' Jeff Williams',
			 'Ronald Wayne','Apple Park','Silicon Valley', 'Apple watch','Apple pay',
			 ' IOS ','Safari','iTunes','Big Tech','Tech Giant','Big Four','Four Horsemen',
			 'Big Five','S&P 5','AAPL','Apple TV','Macintosh','Siri','Shazam'],
	'FB': ['FaceBook','Mark Zuckerberg','Eduardo Saverin','Sheryl Sandberg','Messenger',
		   'Instagram',' FB ',' IG ','WhatsApp','InstaStory','Facebook Dating','Oculus',
		   'Giphy','MapiLLary', 'Menlo Park','Silicon Valley','Big Tech','Tech Giant',
		   'Big Four','Four Horsemen','Big Five','S&P 5'],
	'TWTR': ['Twitter','Tweet','Jack Dorsey','Noah Glass','Biz Stone',
			 'Evan Williams', 'Omid Kordestani','Ned Segal','Parag Agrawal',
			 'TweetDeck',' Vine ','Periscope','MoPub','TWTR'],
	'GOOG': ['Google','Alphabet','Silicon Valley','Big Tech','Tech Giant','Big Four',
			 'Four Horsemen','Big Five','S&P 5','Googleplex','Larry Page','Sergey Brin',
			 'John Hennessy','Sundar Pichai', 'Ruth Porat','DeepMind','Chrome',
			 'Youtube',' YT ','TensorFlow','Android','Nexus'],
}

def preprocess_raw_datasets(folder_path: str, output_path) -> None:
    files_names = glob.glob(folder_path+'/*.csv')
    grouped_datasets = group_datasets(files_names)
    # print(grouped_datasets['AAPL'])
    
	

def group_datasets(files_names: list) -> dict:
    combined_dfs = {}
    columns = ['Text', 'Date', 'Nick', 'Shares', 'Likes']
    
    # Filtering by keywords for each company
    for filename in files_names:
        tmp_df = pd.read_csv(filename)
        tmp_df.drop(columns=['Id'], inplace=True)
        for company, keywords in companies_keywords.items():
            tmp_mask = tmp_df.Text.apply(lambda content: create_mask(content, keywords))
            filtered = tmp_df[tmp_mask]
            
            current = combined_dfs.get(company, pd.DataFrame(columns=columns))
            combined_dfs[company] = pd.concat([current, filtered], ignore_index=True)
            del tmp_mask, current, filtered
        del tmp_df
    
    for k, v in combined_dfs.items():
        v.Text = v.Text.apply(lambda x: " ".join(re.sub("([^0-9A-Za-z \t])|(\w+://\S+)", "", x).split()))
        v.Date = v.Date.apply(lambda x: pd.to_datetime(x.split(' ')[0]))
        v.sort_values(by='Date', inplace=True)
        # msg = ' - {} = {}'.format(k, v.shape)
        # logging.info(msg)
                
    return combined_dfs
    

def create_mask(content: str, keywords: list) -> bool:
	content_ = content.lower()
	keywords_ = [kw.lower() for kw in keywords]
	return any(item for item in keywords_ if item in content_)


def combine_datasets(grouped_companies: dict) -> dict:
    
    pass
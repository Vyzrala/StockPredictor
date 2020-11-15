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
    combined_dfs = None
    
    

def create_mask(content: str, keywords: list) -> bool:
	content_ = content.lower()
	keywords_ = [kw.lower() for kw in keywords]
	return any(item for item in keywords_ if item in content_)
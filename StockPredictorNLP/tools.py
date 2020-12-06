from typing import Dict, List, Tuple
import pandas as pd
import glob, os, logging, datetime, re, copy
import pandas_datareader as pdr
from textblob import TextBlob
from pathlib import Path
import time


class NLPError(Exception): pass


DATE_CUTOFF = '2020-07-14'
COMPANIES_KEYWORDS = {
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
SETTINGS = {
    'format': '%(asctime)s | %(levelname)s | %(funcName)s | %(lineno)s | %(message)s',
    'log_file': '/tools.log',
    'log_folder': os.getcwd()+'/log',
}
Path(SETTINGS['log_folder']).mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=SETTINGS['log_folder'] + SETTINGS['log_file'], 
    filemode='a',
    format=SETTINGS['format'],
    level=logging.INFO
)


def preprocess_raw_datasets(all_tweets: pd.DataFrame, yahoo_data: dict) -> Dict[str, pd.DataFrame]:
    if not all(k in COMPANIES_KEYWORDS.keys() for k in yahoo_data.keys()):
        raise NLPError('Keys in yahoo_data do not match with companies')
    
    all_tweets.drop(columns=['Id'], inplace=True)
    tweets_by_company = get_tweets_by_company(all_tweets)
    sentimeted_data = sentiment_stock_combine(tweets_by_company, yahoo_data)
    
    return sentimeted_data


def get_tweets_by_company(all_tweets: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    print("-> Filtering and grouping tweets dataframe...")
    t0 = time.time()
    combined_dfs = {}
    columns = ['Text', 'Date', 'Nick', 'Shares', 'Likes']
    
    for company, keywords in COMPANIES_KEYWORDS.items():
        tmp_mask = all_tweets.Text.apply(lambda content: create_mask(content, keywords))
        filtered = all_tweets[tmp_mask]
        
        current = combined_dfs.get(company, pd.DataFrame(columns=columns))
        combined_dfs[company] = pd.concat([current, filtered], ignore_index=True)
        del tmp_mask, current, filtered
    
    for k, v in combined_dfs.items():
        v.Text = v.Text.apply(lambda x: " ".join(re.sub("([^0-9A-Za-z \t])|(\w+://\S+)", "", x).split()))
        v.Date = v.Date.apply(lambda x: x.split(' ')[0])
        v.Likes = pd.to_numeric(v.Likes)
        v.Shares = pd.to_numeric(v.Shares)
        v.sort_values(by='Date', inplace=True)
        msg = '- {} = {}'.format(k, v.shape)
        logging.info(msg)
    
    print('\t(time: {:.3f}s)'.format(time.time() - t0))
    return combined_dfs


def create_mask(content: str, keywords: List[str]) -> List[bool]:
	content_ = content.lower()
	keywords_ = [kw.lower() for kw in keywords]
	return any(item for item in keywords_ if item in content_)


def sentiment_stock_combine(grouped_datasets: Dict[str, pd.DataFrame],
                            yahoo_data: Dict[str, pd.DataFrame]) \
                                -> Dict[str, pd.DataFrame]:
    print('-> Sentiment and stock combining...')
    t0 = time.time()
    combined_datasets = {}
    for company_name, dataset in grouped_datasets.items():
        tmp_df = copy.deepcopy(dataset)
        tmp_df = tmp_df[tmp_df.Date >= DATE_CUTOFF]
        tmp_df['Polarity'] = tmp_df.Text.apply(lambda content: TextBlob(content).polarity)
        tmp_df['Subjectivity'] = tmp_df.Text.apply(lambda content: TextBlob(content).subjectivity)
        tmp_df.drop(columns=['Text', 'Nick'], inplace=True)
        
        by_day = tmp_df.groupby(by='Date').mean()
        by_day.index = pd.to_datetime(by_day.index)
        by_day = combine_fridays(by_day)
        
        stock_data = yahoo_data[company_name]
        if 'Date' in stock_data.columns:
            stock_data.set_index(pd.to_datetime(stock_data.Date), inplace=True)
        
        result_ds = pd.concat([stock_data, by_day], join='inner', axis=1)
        # result_ds.reset_index(inplace=True)
        msg = ' - {}:\tshape = {}'.format(company_name, result_ds.shape)
        logging.info(msg)
        combined_datasets[company_name] = result_ds
  
    del grouped_datasets
    print('\t(time: {:.3f}s)'.format(time.time() - t0))
    return combined_datasets


def combine_fridays(grouped_dataset: pd.DataFrame) -> pd.DataFrame:
    for day in grouped_dataset.index:
        if day.weekday() == 4:  # is friday
            to_mean = [day,]
            saturday = day + datetime.timedelta(days=1)
            sunday = day + datetime.timedelta(days=2)
            if saturday in grouped_dataset.index:
                to_mean.append(saturday)
            if sunday in grouped_dataset.index:
                to_mean.append(sunday)
            
            grouped_dataset.loc[day] = pd.DataFrame([grouped_dataset.loc[idx] for idx in to_mean]).mean()
            grouped_dataset.drop(index=to_mean[1:], inplace=True)
    
    return grouped_dataset

def read_files_and_yahoo(path_to_raw: str) -> tuple:
    print('-> Files reading')
    t0 = time.time()
    path_to_raw += '/*.csv'
    files = glob.glob(path_to_raw)
    dtypes = {
        'Id':'str', 
        'Text': 'str', 
        'Date': 'str', 
        'Nick': 'str', 
        'Shares': 'float', 
        'Likes': 'float',
    }

    all_tweets = pd.DataFrame(columns = dtypes.keys())
    all_tweets = all_tweets.astype(dtypes)

    for file in files:
        tmp = pd.read_csv(file)
        all_tweets = all_tweets.append(tmp)


    all_tweets = all_tweets[all_tweets.Date != 'BillGates']
    all_tweets = all_tweets[all_tweets.Date >= DATE_CUTOFF]
    # all_tweets.Date = pd.to_datetime(all_tweets.Date)

    start_date = all_tweets.Date.min()
    end_date = all_tweets.Date.max()
    print(all_tweets.info())
    yahoo_data = {}
    companies = ['AAPL', 'FB', 'GOOG', 'TWTR']

    print('-> Yahoo download')
    for company in companies:
        yahoo_data[company] = pdr.DataReader(company, 'yahoo', start_date, end_date)
    print('\t(time: {:.3f}s)'.format(time.time() - t0))
    return all_tweets, yahoo_data

def save_to_files(data: pd.DataFrame, path: str):
    for k, v in data.items():
        v.to_csv(path+'/'+k+'.csv')
    
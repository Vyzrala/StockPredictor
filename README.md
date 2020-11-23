# StockPredictor
  
### Thesis project of 4 handsome and masculine men :muscle:  
University: Gdansk University of Technology  
Faculty: Electronics, Telecomunications and Infromatics  
Filed of studies: Data Engineering  
Degree: Bechelor

### Team composition:  
* Witold Bazela
* Patryk Dunajewski  
* Kamil Rutkowski  
* Marcin Hebdzyński
    
# Instalation

> *instalation package is in /distribution/ directory*
```
Windows: pip install StockPredictor-(version).tar.gz
Linux: pip3 install StockPredictor-(version).tar.gz
```
# Example usage

### For LSTM Predictor  
```
from StockPredictorLSTM import PredictorLSTM

company = 'AAPL'
forecasted_value_name = 'Close'
days_forword = 15
start_date = '2015-01-01'
end_date = '2020-01-01'
predictor = PredictorLSTM()

# Initial use
dataset = predictor.download_dataset(start_date, end_date, company)
predictor.create_model(dataset)
predictor.display_info()
predictor.predict(days_forword)
predictor.prediction_plot(forecasted_value_name, company, days_forword)
predictor.save_model(company)

# Further uses
predictor.load_model(company)
predictor.predict(days_forword)
predictor.prediction_plot(forecasted_value_name, company, days_forword)
```

### For NLP Predictor

```
from StockPredictorNLP.tools import preprocess_raw_datasets
from StockPredictorNLP import PredictorNLP

# Data preprocessing
predictor = PredictorNLP()
company_name = 'AAPL'

# all_tweets - pandas DataFrame with all collected tweets
# yahoo_data - dict of stock data downloaded from yahoo finance where key is a company name and value pandas dataframe 

datasets_dict = preprocess_raw_datasets(all_tweets, yahoo_data)

dataset = datasets_dict[company_name]

# Model creation and prediction
predictor.create_model(dataset)
prediction = predictor.predict()

# Saving model
save_folder_path = '/' + company_name  # '/AAPL'
predictor.save_model(save_folder_path)

# Loading saved model and predicting
predictor.load_model(save_folder_path)  
prediction = predictor.predict()
```
# StockPredictor
  
### Thesis project of 4 handsome and masculine men :muscle:  
##
University: Gdansk University of Technology  
Faculty: Electronics, Telecomunications and Infromatics  
Filed of studies: Data Engineering  
Degree: Bechelor
    
# Instalation

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
from StockPredictorNLP import PredictorNLP
from StockPredictorNLP.tools import preprocess_raw_datasets


# When you want to pre-process raw dataset first and then get predictions from that dataset
company_name = 'AAPL'
datasets_folder_path = 'L:/example/path'
preprocessed_files_folder_name = 'Pre-processed data'
predictor = PredictorNLP()

# Get preprocessed datasets
datasets_dict, abs_output_path = preprocess_raw_datasets(datasets_folder_path, preprocessed_files_folder_name)

# If you do not want to read preprocessed data from file and just insert dataset into Predictor classes
predictor.create_model(datasets_dict[company_name])
prediction = predictor.predict() 


# If you want to specify path to dataset there are 2 approaches:
# 1 (More automative approach)
company_dataset_file_path = abs_output_path+'/'+company_name+'.csv'
# or 2 (Just provide absolute path, this is good when you already have preprocessed data) 
company_dataset_file_path = ' D:/Programing projects/StockPredictor/Pre-processed data/AAPL.csv'
#  - Those paths are the same 
dataset = predictor.get_data_from_file(company_dataset_file_path)
# Disclimer: those datasets are the same: dataset[company_name] == dataset

# Then just
predictor.create_model(dataset)
prediction = predictor.predict() 


```


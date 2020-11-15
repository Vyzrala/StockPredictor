# StockPredictor
  
### Thesis project of 4 handsome and masculine men :muscle:  
##
University: Gdansk University of Technology  
Faculty: Electronics, Telecomunications and Infromatics  
Filed of studies: Data Engineering  
Degree: Bechelor
    
# Instalation

```
Windows: pip install StockPredictoLSTM-(version).tar.gz
Linux: pip3 install StockPredictoLSTM-(version).tar.gz
```
# Example usage

### For LSTM Predictor  
```
from StockPredictorLSTM import Predictor

company = 'AAPL'
forecasted_value_name = 'Close'
days_forword = 15
start_date = '2015-01-01'
end_date = '2020-01-01'
predictor = Predictor()

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
TODO: 
```


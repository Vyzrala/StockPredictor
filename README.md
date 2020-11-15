# StockPredictor
  
### Thesis project of 4 handsome and masculine men :muscle:  
##
University: Gdansk University of Technology  
Faculty: Electronics, Telecomunications and Infromatics  
Filed of studies: Data Engineering  
Degree: Bechelor
  
<!---
Installable package and manual are in *.zip file under 'StockPredictorLSTM/distribution' directory.
#### Package creation tutorials:
    1. https://betterscientificsoftware.github.io/python-for-hpc/tutorials/python-pypi-packaging/
    2. https://pythonhosted.org/an_example_pypi_project/setuptools.html
    3. https://packaging.python.org/guides/distributing-packages-using-setuptools/
--->
  
# Instalation

```
Windows:  double click on *.exe from distribution/StockPredictorLSTM.zip file or pip install StockPredictoLSTM-(version).tar.gz
Linux: pip3 install StockPredictoLSTM-(version).tar.gz
```
/////////////
# Example use 

```
from StockPredictorLSTM import Predictor

company = 'AAPL'
forecasted_value_name = 'Close'
days_forword = 15
start_date = '2015-01-01'
end_date = '2020-01-01'
predictor = Predictor()

# Use 1 - Initial use
dataset = predictor.download_dataset(start_date, end_date, company)
predictor.create_model(dataset)
predictor.display_info()
predictor.predict(days_forword)
predictor.prediction_plot(forecasted_value_name, company, days_forword)
predictor.save_model(company)

# Use 2
predictor.load_model(company)
predictor.predict(days_forword)
predictor.prediction_plot(forecasted_value_name, company, days_forword)

```
<!---
# Credits

Creators:
> Witold Bazela  
> Patryk Dunajewski  
> Marcin Hebdzyński  
> Kamil Rutkowski
--->

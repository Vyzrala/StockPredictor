import pandas as pd
import numpy as np
import os


class PredictorNLP:
    def __init__(self) -> None:
        pass
    
    def create_model(self, dataset: pd.DataFrame):
        pass
    
    def predict(self):
        pass
    
    def save_model(self, path: str):
        pass
    
    def load_model(self, path: str):
        pass

    def get_data_from_file(self, path_to_file):
        dataset = None
        if os.path.isfile(path_to_file):
            dataset = pd.read_csv(path_to_file)  
            return dataset          
        else:
            raise NLPError('\nNo such file.\nPlease check if file at {} exists.'.format(path_to_file))


class NLPError(Exception): pass
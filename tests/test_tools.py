import unittest
# import mock
import os, glob, shutil

from StockPredictorNLP.tools import preprocess_raw_datasets


class TestTools(unittest.TestCase):
    def test_Preprocess_raw_datasets(self):
        test_files_path = '/home/marcin/Documents/python_projects/StockPredictor/StockPredictor/tests/test_data'
        output_path = 'tests/tests_output'
        
        result = preprocess_raw_datasets(test_files_path, output_path)
        values_bool = any([False if v.empty else True for v in result.values()])
        keys_bool = all(k for k in result.keys() if type(k) == str)
        self.assertTrue(values_bool)
        self.assertTrue(keys_bool)
        
        glob_path = './'+output_path
        self.assertTrue(os.path.exists(glob_path))  # Check wether output_path was created
        
        files = glob.glob(glob_path+'/*')
        self.assertEqual(len(files), 4)  # Check wether function produced 4 files
        shutil.rmtree(glob_path+'/')
    
    # def test_Preprocess_raw_datasets_mock(self):
        
    #     pass
        
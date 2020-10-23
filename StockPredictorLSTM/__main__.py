from StockPredictorLSTM import Predictor
import pandas_datareader as pdr
import datetime

if __name__ == "__main__":
    pass
    COMPANY_NAME = 'FB'
    START_DATE = '2017-01-01'
    END_DATE = '2019-05-02'#str(datetime.datetime.today().date())
    SOURCE = 'yahoo'
    df = pdr.DataReader(COMPANY_NAME, SOURCE, START_DATE, END_DATE)
    df.reset_index(inplace=True)

    model = Predictor()
    model.load_model(COMPANY_NAME)
    model.change_dataset(df)
    # model.create_model(df)
    model.display_info()
    days = 15
    print("\n{} days forword:\n".format(days), model.predict(days))

    ### Saving model test
    # model.save_model(COMPANY_NAME)
    
    ### Loading model test

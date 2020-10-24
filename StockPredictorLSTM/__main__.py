from StockPredictorLSTM import Predictor
import datetime

if __name__ == "__main__":
    COMPANY_NAME = 'FB'
    START_DATE = '2015-01-01'
    END_DATE = str(datetime.datetime.today().date())

    model = Predictor()
    model.load_model(COMPANY_NAME)
    # model.create_model(model.download_dataset(START_DATE, END_DATE, COMPANY_NAME))
    model.change_dataset(model.download_dataset('2019-05-02', END_DATE, COMPANY_NAME))
    model.display_info()
    days = 15
    print("\n{} days forword:\n".format(days), model.predict(days))
    model.prediction_plot('Close', COMPANY_NAME, days)

    # model.save_model(COMPANY_NAME)

    del model
    
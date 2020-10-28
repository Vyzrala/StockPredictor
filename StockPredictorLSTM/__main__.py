from StockPredictorLSTM import Predictor
import datetime

if __name__ == "__main__":
    COMPANY_NAME = 'FB'
    START_DATE = '2015-01-01'
    END_DATE = str(datetime.datetime.today().date()) # '2020-01-01'

    model = Predictor()
    model.load_model(COMPANY_NAME)
    # model.create_model(model.download_dataset(START_DATE, END_DATE, COMPANY_NAME))
    # df = model.download_dataset('2019-05-02', END_DATE, COMPANY_NAME)
    # model.change_dataset(df)
    model.display_info()
    days = 15
    predictions = model.predict(days)
    print("\n{} days forword:\n".format(days), predictions)
    model.prediction_plot('Close', COMPANY_NAME, days)
    
    # for feature in list(predictions.columns[1:]):  # Fake mesure
    #     print("{}:\t{}".format(feature, model.compare_directions(predictions, df[-days:], feature)))
    # model.save_model(COMPANY_NAME)


    del model
    
from StockPredictorLSTM import Predictor
import datetime

if __name__ == "__main__":
    COMPANY_NAME = 'AAPL'
    START_DATE = '2012-01-01'
    END_DATE = '2018-07-01' #str(datetime.datetime.today().date()) # '2020-01-01'

    model = Predictor()
    # model.load_model(COMPANY_NAME)
    # df = model.download_dataset(START_DATE, END_DATE, COMPANY_NAME)
    # model.change_dataset(df)
    
    model.create_model(model.download_dataset(START_DATE, END_DATE, COMPANY_NAME))
    model.display_info()
    days = 15
    predictions = model.predict(days)
    print("\n{} days forword:\n".format(days), predictions)
    model.prediction_plot('Close', COMPANY_NAME, days)
    
    model.save_model(COMPANY_NAME)
    # compare_df = model.download_dataset(END_DATE, str(datetime.datetime.strptime(END_DATE, "%Y-%m-%d").date() + datetime.timedelta(days=days)), COMPANY_NAME)
    # for feature in list(predictions.columns[1:]):  # Fake mesure
    #     print("{}:\t{}".format(feature, model.compare_directions(predictions, compare_df[-days:], feature)))


    del model
    
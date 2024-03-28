import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
from tensorflow.keras.models import load_model
import joblib


# def test(today, num_of_days, data, model, scaler, sequence_length) -> int:
    
#     specific_date = today  # Adjust this date as needed

#     # Extract the data for the specific date and the preceding SEQUENCE_LENGTH - 1 dates
#     specific_date_index = data.index.get_loc(specific_date)

#     num_of_correct_predictions = 0

#     for x in range(num_of_days):

#         input_features = data["Differenced Closing"].iloc[specific_date_index - sequence_length + 1 : specific_date_index + 1].values
        
#         # Reshape the input features to match the input shape expected by the LSTM model
#         input_features_reshaped = input_features.reshape(1, sequence_length, 1)  # Assuming one feature dimension

#         input_features_scaled = scaler.transform(input_features_reshaped.reshape(-1, 1))
#         input_features_scaled = input_features_scaled.reshape(1, sequence_length, 1)

#         # Make predictions using the model
#         prediction = model.predict(input_features_scaled)

#         specific_date_index += 1

#     return num_of_correct_predictions

def test(data, model, scaler, sequence_length):
    # Get the index of the last date in the DataFrame
    today_index = len(data) - 1
    
    # Extract the data for the specific date and the preceding SEQUENCE_LENGTH - 1 dates
    input_features = data["Differenced Closing"].iloc[today_index - sequence_length + 1 : today_index + 1].values
    print(input_features)
    input_features_reshaped = input_features.reshape(1, sequence_length, 1)  # Assuming one feature dimension

    input_features_scaled = scaler.transform(input_features_reshaped.reshape(-1, 1))
    input_features_scaled = input_features_scaled.reshape(1, sequence_length, 1)
    # Make the prediction using the model
    prediction = model.predict(input_features_reshaped)

    return prediction.tolist()


def is_row_today(pandas_row) -> bool:
    today = datetime.now().date()
    row_date = pandas_row.date[0]
    
    return today <= row_date


def main():

    stock = yf.Ticker("SHB-A.ST")

    data = stock.history(period="3mo")

    data["Differenced Closing"] = data["Close"].diff()
    last_date_in_stock_history = data.tail(1).index
    print(data.to_string())

    # if is_row_today(last_date_in_stock_history):
    #     data.drop(data.tail(1).index, inplace=True)


    model = load_model("/Users/erikolofsson/Documents/Kodning/Python/StockPricePredictor/App/Scripts/Models/my_model.keras")
    scaler = joblib.load("/Users/erikolofsson/Documents/Kodning/Python/StockPricePredictor/App/Scripts/Models/my_scaler.pkl")

    X = np.linspace(-3, 3, 20)
    predictions = []

     
    for i in X:

        
        print(data.iloc[-52, data.columns.get_loc("Close")])
        
        data.iloc[-30, data.columns.get_loc("Differenced Closing")] = i # Change value of todays change in closing price
        predictions.append(test(data, model, scaler, 5))


    for x in predictions:
        print(x)
    

if __name__ == '__main__':
    main()
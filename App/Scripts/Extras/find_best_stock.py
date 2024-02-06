import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import os

DIRECTORY = "Raw_data"

def run(file_path):
    column_order = ['Date', 'Bid', 'Ask', 'Opening price', 'High price', 'Low price', 'Closing price', 'Average price', 'Total volume', 'Turnover', 'Trades']
    data = pd.read_csv(file_path, sep=";", decimal=",", skiprows=1, usecols=column_order, encoding="utf8")
    pd.options.mode.chained_assignment = None  # default='warn'

    columns_to_drop = ["Bid", "Ask", "Opening price", "Average price"]
    data = data.set_index("Date").drop(columns=columns_to_drop)

    data = data.dropna(axis=1)

    # Change order of rows oldest to newest
    data = data[::-1]

    data["Differenced Closing"] = data["Closing price"].diff()


    SHORT_CHANGE_HORIZON = 5
    MID_CHANGE_HORIZON = 20
    LONG_CHANGE_HORIZON = 50
    FACTOR = 0.014 # Profit to define target 0.014

    data["Short_day_change"] = (data["Closing price"] - data["Closing price"].shift(SHORT_CHANGE_HORIZON)) / SHORT_CHANGE_HORIZON
    data["Mid_day_change"] = (data["Closing price"] - data["Closing price"].shift(MID_CHANGE_HORIZON)) / MID_CHANGE_HORIZON
    data["Long_day_change"] = (data["Closing price"] - data["Closing price"].shift(LONG_CHANGE_HORIZON)) / LONG_CHANGE_HORIZON


    data["t+1"] = data["Closing price"].shift(-1)
    data["t+2"] = data["Closing price"].shift(-2)
    data["t+3"] = data["Closing price"].shift(-3)
    data["t+4"] = data["Closing price"].shift(-4)
    data["t+5"] = data["Closing price"].shift(-5)

    conditions = [
        (data["t+1"] > (data["Closing price"] * (1+FACTOR))) |
        (data["t+2"] > (data["Closing price"] * (1+FACTOR))) |
        (data["t+3"] > (data["Closing price"] * (1+FACTOR))) |
        (data["t+4"] > (data["Closing price"] * (1+FACTOR))) |
        (data["t+5"] > (data["Closing price"] * (1+FACTOR)))
    ]

    choice = [1]
    data["Target"] = np.select(conditions, choice, 0)

    data = data.dropna()


    TRAIN_RATIO = 0.8
    split_ix = int(len(data) * TRAIN_RATIO)

    # Split the data into training and testing sets
    x = data[["Differenced Closing", "Short_day_change", "Mid_day_change", "Long_day_change"]].values
    y = data["Target"].values

    X_train, X_test = x[:split_ix], x[split_ix:]
    y_train, y_test = y[:split_ix], y[split_ix:]

    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
    model.fit(X_train, y_train)

    y_pred = predict(X_train=X_train, y_train=y_train, X_test=X_test, model=model, threshold=0.55)
    
    print("ROC: " + str(print_report(y_pred, y_test, model)))
    return accuracy_score(y_test, y_pred)



def print_report(pred, y_test, model):

    accuracy = accuracy_score(y_test, pred)
    print(f'Accuracy: {accuracy:.2f}')
    classification_report_str = classification_report(y_test, pred)
    print('Classification Report:\n', classification_report_str)

    # Calculate the AUC (Area Under the ROC Curve)
    fpr, tpr, thresholds = roc_curve(y_test, pred)
    roc_auc = auc(fpr, tpr)
    return roc_auc

    

def predict(X_train, y_train, X_test, model, threshold):
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:,1]
    
    preds[preds >= threshold] = 1
    preds[preds < threshold] = 0

    return preds



def main():
    files = os.listdir(DIRECTORY)
    stocks = {}
    
    for x in files:
        if x == ".DS_Store":
            continue
    # #     # stocks[x.split('-')[0]] = run("Raw_data/" + x)
        print(x)
        stocks[x] = run(DIRECTORY + '/' + x)  
        print("----------")
    
    for x in stocks:
        print(x)
        print(stocks[x])
        print("-----")



if __name__ == "__main__":
    main()
    
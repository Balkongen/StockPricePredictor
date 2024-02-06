import joblib

model = joblib.load("Prediction/model_RF.pkl")

# Difference in price between t and t-1
change = -0.68
five_day_change = 0.04 # Not including today closing price

input_data = [[change, five_day_change]]

prediction = model.predict(input_data)

if prediction == 1:
    print("Increase")
else:
    print("Decrease")

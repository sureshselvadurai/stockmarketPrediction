import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

df = pd.read_csv(f"data/raw/apple_processed.csv")

# Use only relevant columns for modeling
features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'lag_1']
target = 'target_scaled'
df = df[['Timestamp'] + features + [target]]

# Drop rows with missing values
df = df.dropna()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df[features + [target]])

# Create lagged feature 'lag_1'
df_scaled = pd.DataFrame(df_scaled, columns=features + [target])
df_scaled['lag_1'] = df_scaled[target].shift(1)
df_scaled = df_scaled.dropna()

# Split the data into training and testing sets
train_size = int(len(df_scaled) * 0.8)
train, test = df_scaled[0:train_size], df_scaled[train_size:len(df_scaled)]


# Convert the data to the format expected by LSTM (numpy arrays)
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, -1])
    return np.array(dataX), np.array(dataY)


look_back = 3  # Number of time steps to look back
trainX, trainY = create_dataset(train.values, look_back)
testX, testY = create_dataset(test.values, look_back)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, len(features) + 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(trainX, trainY, epochs=120, batch_size=1, verbose=2)

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Invert predictions to original scale
trainPredict = scaler.inverse_transform(np.hstack((trainX[:, -1, :-1], trainPredict.reshape(-1, 1))))
trainY = scaler.inverse_transform(np.hstack((trainX[:, -1, :-1], trainY.reshape(-1, 1))))

testPredict = scaler.inverse_transform(np.hstack((testX[:, -1, :-1], testPredict.reshape(-1, 1))))
testY = scaler.inverse_transform(np.hstack((testX[:, -1, :-1], testY.reshape(-1, 1))))

# Calculate RMSE
trainScore = np.sqrt(mean_squared_error(trainY[:, -1], trainPredict[:, -1]))
testScore = np.sqrt(mean_squared_error(testY[:, -1], testPredict[:, -1]))

# Create a DataFrame to store the results
columns = ['Tag', 'Prediction', 'Features', 'Time']
train_df = pd.DataFrame({'Tag': 'Train',
                         'Prediction': trainPredict[:, -1],
                         'Features': trainX[:, -1, :-1].tolist(),
                         'Time': df['Timestamp'][look_back - 1:look_back - 1 + len(trainY)]})

test_df = pd.DataFrame({'Tag': 'Test',
                        'Prediction': testPredict[:, -1],
                        'Features': testX[:, -1, :-1].tolist(),
                        'Time': df['Timestamp'][look_back - 1 + len(trainY) + look_back - 1:look_back - 1 + len(trainY) + look_back - 1 + len(testY)]})

# Concatenate train and test DataFrames
result_df = pd.concat([train_df, test_df])

# Save the DataFrame to a CSV file
result_df.to_csv(f"data/raw/apple_prediction.csv", index=False)


# Plot the results
plt.figure(figsize=(14, 6))

# Plot training data
plt.plot(df['Timestamp'][look_back - 1:look_back - 1 + len(trainY)], trainY[:, -1], label='Actual (Train)', color='blue')
plt.plot(df['Timestamp'][look_back - 1:look_back - 1 + len(trainY)], trainPredict[:, -1], label='Predicted (Train)', color='red')

# Plot testing data
plt.plot(df['Timestamp'][look_back - 1 + len(trainY) + look_back - 1:look_back - 1 + len(trainY) + look_back - 1 + len(testY)], testY[:, -1], label='Actual (Test)', color='green')
plt.plot(df['Timestamp'][look_back - 1 + len(trainY) + look_back - 1:look_back - 1 + len(trainY) + look_back - 1 + len(testY)], testPredict[:, -1], label='Predicted (Test)', color='orange')

plt.title('LSTM Model for Stock Price Prediction')
plt.xlabel('Timestamp')
plt.ylabel('Scaled Stock Price')
plt.legend()
plt.show()

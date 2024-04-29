import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

class StockRequest(BaseModel):
    stock_name: str

# Define file paths (assuming the structure is already known)
STOCK_FILE_PATHS = {
    "TSLA": "E:/Data Science/Stock Price Prediction/Stock Price Prediction/data/TSLA_data.csv",
    "AAPL": "E:/Data Science/Stock Price Prediction/Stock Price Prediction/data/AAPL_data.csv",
    "AMZN": "E:/Data Science/Stock Price Prediction/Stock Price Prediction/data/AMZN_data.csv",
    "MSFT": "E:/Data Science/Stock Price Prediction/Stock Price Prediction/data/MSFT_data.csv",
}

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['Close'] = df['Close'].astype(float)  # Ensure data is float
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

def prepare_datasets(data, seq_len):
    x, y = [], []
    for i in range(seq_len, len(data)):
        x.append(data[i-seq_len:i, 0])
        y.append(data[i, 0])
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x, y

@app.post("/LSTM_Predict")
async def predict(stock_request: StockRequest):
    stock_name = stock_request.stock_name
    if stock_name not in STOCK_FILE_PATHS:
        raise HTTPException(status_code=422, detail="Invalid Stock Name")

    filepath = STOCK_FILE_PATHS[stock_name]
    data, scaler = load_and_preprocess_data(filepath)
    seq_len = 60  # Sequence length for LSTM
    x_train, y_train = prepare_datasets(data[:int(0.8 * len(data))], seq_len)
    x_test = prepare_datasets(data[int(0.8 * len(data)):], seq_len)[0]  # Only x_test needed

    # Define and compile the model
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fit model
    model.fit(x_train, y_train, epochs=5, batch_size=32)

    # Predicting
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)  # Inverse transform to get actual values

    predict_prices = predictions.flatten().tolist()  # Convert to list for JSON response
    return {"prediction": predict_prices}


import datetime
from pathlib import Path

import joblib
import pandas as pd
import yfinance as yf
from prophet import Prophet

import argparse

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from model import predict, convert

BASE_DIR = Path(__file__).resolve(strict=True).parent
TODAY = datetime.date.today()

app = FastAPI()

def train(ticker="MSFT"):
    data = yf.download(ticker, "2020-01-01", TODAY.strftime("%Y-%m-%d"))

    df_forecast = data.copy()
    df_forecast.reset_index(inplace=True)
    df_forecast["ds"] = df_forecast["Date"]
    df_forecast["y"] = df_forecast["Adj Close"]
    df_forecast = df_forecast[["ds", "y"]]
    df_forecast

    model = Prophet()
    model.fit(df_forecast)

    joblib.dump(model, Path(BASE_DIR).joinpath(f"{ticker}.joblib"))


def predict(ticker="MSFT", days=7):
    model_file = Path(BASE_DIR).joinpath(f"{ticker}.joblib")
    if not model_file.exists():
        return False

    model = joblib.load(model_file)

    future = TODAY + datetime.timedelta(days=days)

    dates = pd.date_range(start="2020-01-01", end=future.strftime("%m/%d/%Y"), )
    df = pd.DataFrame({"ds": dates})

    forecast = model.predict(df)

    model.plot(forecast).savefig(f"{ticker}_plot.png")
    model.plot_components(forecast).savefig(f"{ticker}_plot_components.png")

    return forecast.tail(days).to_dict("records")


def convert(prediction_list):
    output = {}
    for data in prediction_list:
        date = data["ds"].strftime("%m/%d/%Y")
        output[date] = data["trend"]
    return output

app = FastAPI()

# pydantic models
class StockIn(BaseModel):
    ticker: str
    days: int

class StockOut(StockIn):
    forecast: dict

@app.post("/predict", response_model=StockOut, status_code=200)
def get_prediction(payload: StockIn):
    ticker = payload.ticker
    days = payload.days

    prediction_list = predict(ticker, days)

    if not prediction_list:
        raise HTTPException(status_code=400, detail="Model not found.")

    response_object = {
        "ticker": ticker,
        "days": days,
        "forecast": convert(prediction_list)}
    return response_object

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('--ticker', type=str, default='MSFT', help='Stock Ticker')
    parser.add_argument('--days', type=int, default=7, help='Number of days to predict')
    args = parser.parse_args()

    train(args.ticker)
    prediction_list = predict(ticker=args.ticker, days=args.days)
    output = convert(prediction_list)
    print(output)
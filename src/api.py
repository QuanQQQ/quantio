from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os

from .database import get_all_stocks, get_stock_daily

ROOT = os.path.dirname(os.path.dirname(__file__))

app = FastAPI(title="Quantio API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StockItem(BaseModel):
    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    listing_date: Optional[str] = None

class KlineBar(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float

@app.get("/api/stocks", response_model=List[StockItem])
def stocks():
    df = get_all_stocks(filter_tradable=False)
    return df.to_dict(orient="records")

@app.get("/api/kline", response_model=List[KlineBar])
def kline(symbol: str = Query(...), start: Optional[str] = None, end: Optional[str] = None):
    df = get_stock_daily(symbol, start, end)
    if df.empty:
        return []
    # 只返回所需字段
    df = df[["date", "open", "high", "low", "close", "volume"]]
    return df.to_dict(orient="records")

@app.get("/api/trades")
def trades():
    path = os.path.join(ROOT, "backtest_trades_dynamic.csv")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

@app.get("/api/equity")
def equity():
    path = os.path.join(ROOT, "backtest_equity_curve.csv")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

@app.get("/api/operations")
def operations():
    path = os.path.join(ROOT, "backtest_operations.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

def run():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    run()

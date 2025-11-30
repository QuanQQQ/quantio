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

class DailyBar(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float

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

@app.get("/api/daily", response_model=List[DailyBar])
def daily(symbol: str = Query(...), start: Optional[str] = None, end: Optional[str] = None):
    df = get_stock_daily(symbol, start, end)
    if df.empty:
        return []
    df = df[["date", "open", "high", "low", "close", "volume", "amount"]]
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

@app.get("/api/db/stats")
def db_stats():
    import sqlite3
    conn = sqlite3.connect(os.path.join(ROOT, "data", "stock_data.db"))
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM daily_prices")
        rows = cur.fetchone()[0]
        cur.execute("SELECT MIN(date), MAX(date) FROM daily_prices")
        mn, mx = cur.fetchone()
        cur.execute("SELECT COUNT(DISTINCT symbol) FROM daily_prices")
        symbols_in_prices = cur.fetchone()[0]
        # stocks table
        cur.execute("SELECT COUNT(*) FROM stocks")
        stocks_count = cur.fetchone()[0]
        return {
            "rows": rows,
            "min_date": mn,
            "max_date": mx,
            "symbols_in_prices": symbols_in_prices,
            "stocks_count": stocks_count,
        }
    finally:
        conn.close()

def run():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    run()

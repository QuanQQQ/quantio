import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os

from database import get_all_stocks, get_stock_daily

st.set_page_config(layout="wide", page_title="回测可视化")

ROOT = os.path.dirname(os.path.dirname(__file__))

@st.cache_data
def load_equity():
    path = os.path.join(ROOT, "backtest_equity_curve.csv")
    df = pd.read_csv(path, dtype={"date": str})
    return df

@st.cache_data
def load_trades():
    path = os.path.join(ROOT, "backtest_trades_dynamic.csv")
    df = pd.read_csv(path, dtype={
        "symbol": str,
        "entry_date": str,
        "exit_date": str
    })
    return df

@st.cache_data
def load_operations():
    path = os.path.join(ROOT, "backtest_operations.csv")
    df = pd.read_csv(path, dtype={
        "date": str,
        "action": str,
        "symbol": str
    })
    return df

def calc_max_drawdown(equity_series):
    peak = -float('inf')
    mdd = 0.0
    for v in equity_series:
        peak = max(peak, v)
        if peak > 0:
            dd = v / peak - 1
            if dd < mdd:
                mdd = dd
    return mdd

def calc_annual_return(equity_series):
    if len(equity_series) == 0:
        return 0.0
    start = equity_series.iloc[0]
    end = equity_series.iloc[-1]
    days = len(equity_series)
    if start <= 0 or days <= 0:
        return 0.0
    return (end / start) ** (365 / days) - 1

def scale_equity(equity_series, principal):
    if len(equity_series) == 0:
        return equity_series
    base = equity_series.iloc[0]
    if base == 0:
        return equity_series
    scale = principal / base
    return equity_series * scale

def main():
    st.title("📊 回测数据可视化")

    # Sidebar inputs
    st.sidebar.header("参数")
    principal = st.sidebar.number_input("本金", value=100000, min_value=1000, step=1000)

    # Load data
    try:
        equity_df = load_equity()
        trades_df = load_trades()
        ops_df = load_operations()
    except Exception as e:
        st.error(f"加载 CSV 失败: {e}")
        return

    # Date range
    try:
        equity_df['date_dt'] = pd.to_datetime(equity_df['date'], format="%Y%m%d")
        min_date = equity_df['date_dt'].min()
        max_date = equity_df['date_dt'].max()
    except Exception:
        min_date = datetime.now() - timedelta(days=180)
        max_date = datetime.now()

    date_range = st.sidebar.date_input("时间范围", value=(min_date, max_date))

    # Stocks for search (prefer symbols in trades; fallback to DB)
    trade_symbols = sorted(trades_df['symbol'].unique().tolist()) if not trades_df.empty else []
    op_symbols = sorted(ops_df['symbol'].unique().tolist()) if not ops_df.empty else []
    stocks_df = pd.DataFrame()
    try:
        stocks_df = get_all_stocks(filter_tradable=False)
    except Exception:
        pass
    all_symbols = op_symbols or trade_symbols or (stocks_df['symbol'].tolist() if not stocks_df.empty else [])

    selected_symbol = st.sidebar.selectbox("股票代码", options=["(不选)"] + all_symbols, index=0)
    selected_symbol = None if selected_symbol == "(不选)" else selected_symbol

    st.sidebar.markdown("---")

    # Layout
    col_left, col_right = st.columns(2)

    # Right: Equity and indicators
    with col_right:
        if equity_df.empty:
            st.warning("净值数据为空")
        else:
            # Filter by date
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                eq = equity_df[(equity_df['date_dt'] >= start_dt) & (equity_df['date_dt'] <= end_dt)].copy()
            else:
                eq = equity_df.copy()

            eq_scaled = scale_equity(eq['equity'], principal)
            mdd = calc_max_drawdown(eq_scaled)
            annual = calc_annual_return(eq_scaled)
            total_ret = (eq_scaled.iloc[-1] / eq_scaled.iloc[0] - 1) if len(eq_scaled) >= 2 else 0.0

            mi1, mi2, mi3 = st.columns(3)
            mi1.metric("最大回撤", f"{mdd*100:.2f}%")
            mi2.metric("年化收益", f"{annual*100:.2f}%")
            mi3.metric("累计收益", f"{total_ret*100:.2f}%")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eq['date_dt'], y=eq_scaled, name='账户净值', mode='lines'))
            fig.update_layout(height=360, xaxis_title='日期', yaxis_title='净值', dragmode='pan')
            st.plotly_chart(fig, use_container_width=True)

    # Left: Kline with trades
    with col_left:
        if selected_symbol:
            try:
                start_str = pd.to_datetime(date_range[0]).strftime("%Y%m%d") if isinstance(date_range, (list, tuple)) else None
                end_str = pd.to_datetime(date_range[1]).strftime("%Y%m%d") if isinstance(date_range, (list, tuple)) else None
                kdf = get_stock_daily(selected_symbol, start_str, end_str)
            except Exception as e:
                st.error(f"加载 K 线失败: {e}")
                kdf = pd.DataFrame()

            if kdf.empty:
                st.info("无 K 线数据或未选择时间范围")
            else:
                kdf['date_dt'] = pd.to_datetime(kdf['date'])
                # Filter operations by symbol and range
                odf = ops_df[ops_df['symbol'] == selected_symbol].copy()
                odf['date_dt'] = pd.to_datetime(odf['date'], format="%Y%m%d", errors='coerce')
                if isinstance(date_range, (list, tuple)):
                    odf = odf[(odf['date_dt'] >= pd.to_datetime(date_range[0])) & (odf['date_dt'] <= pd.to_datetime(date_range[1]))]

                # Build figure with volume
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.8])
                fig.add_trace(go.Candlestick(
                    x=kdf['date_dt'], open=kdf['open'], high=kdf['high'], low=kdf['low'], close=kdf['close'], name='价格'
                ), row=1, col=1)
                fig.add_trace(go.Bar(x=kdf['date_dt'], y=kdf['volume'], name='成交量'), row=2, col=1)

                # Markers for trades
                buy_scatter = go.Scatter(
                    x=odf[odf['action']=='buy']['date_dt'],
                    y=odf[odf['action']=='buy']['price'],
                    mode='markers', name='买入',
                    marker=dict(color='green', symbol='triangle-up', size=10)
                )
                sell_scatter = go.Scatter(
                    x=odf[odf['action']=='sell']['date_dt'],
                    y=odf[odf['action']=='sell']['price'],
                    mode='markers', name='卖出',
                    marker=dict(color='red', symbol='triangle-down', size=10),
                    text=odf[odf['action']=='sell'].apply(lambda r: f"{r.get('reason','')} {("半仓" if (r.get('partial_ratio') and r.get('partial_ratio',1)<1) else "全仓")}", axis=1),
                    hovertemplate='卖出: %{y}<br>%{text}'
                )
                fig.add_trace(buy_scatter, row=1, col=1)
                fig.add_trace(sell_scatter, row=1, col=1)

                fig.update_layout(height=420, xaxis_rangeslider_visible=False, dragmode='pan')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("在左侧选择股票代码以查看 K 线与买卖点")

    # Bottom: Holdings table
    st.markdown("---")
    st.subheader("📋 每日操作明细")

    # Build per-day capital and rows (simple equal-weight per day, matching HTML viewer logic)
    if ops_df.empty:
        st.info("操作数据为空")
    else:
        # Prepare operations
        ops_df['date_dt'] = pd.to_datetime(ops_df['date'], format="%Y%m%d", errors='coerce')
        uniq_dates = sorted(ops_df['date'].unique().tolist())
        sel_date = st.selectbox("筛选日期（操作日）", options=["(全部)"] + uniq_dates, index=0)
        filtered_ops = ops_df if sel_date == "(全部)" else ops_df[ops_df['date'] == sel_date]

        # Join sells with trades to compute realized PnL
        trades_by_key = trades_df.set_index(['symbol','exit_date']) if not trades_df.empty else pd.DataFrame()

        rows = []
        for _, r in filtered_ops.iterrows():
            action = r.get('action','')
            sym = r.get('symbol','')
            dt = r.get('date','')
            price = float(r.get('price',0) or 0)
            qty = float(r.get('quantity',0) or 0)
            value_amt = price * qty
            pnl_amt = None
            reason = r.get('reason','')
            pr = r.get('partial_ratio', None)
            hold_days = r.get('hold_days', None)
            if action == 'sell' and not trades_by_key.empty:
                key = (sym, dt)
                if key in trades_by_key.index:
                    tr = trades_by_key.loc[key]
                    # handle possible multi-index selection
                    if isinstance(tr, pd.DataFrame):
                        tr = tr.iloc[0]
                    entry_price = float(tr.get('entry_price',0) or 0)
                    exit_price = float(tr.get('exit_price',price) or price)
                    sell_qty = float(tr.get('quantity',qty) or qty)
                    pnl_amt = (exit_price - entry_price) * sell_qty
            rows.append({
                '日期': dt,
                '动作': '买入' if action=='buy' else '卖出',
                '股票代码': sym,
                '价格': price,
                '数量': qty,
                '金额(¥)': value_amt,
                '卖出盈亏(¥)': pnl_amt if pnl_amt is not None else '',
                '原因': reason,
                '比例': (f"{int(pr*100)}%" if (pr is not None and pr<1) else ("100%" if action=='sell' else '')),
                '持仓天数': hold_days if action=='sell' else '',
            })
        out_df = pd.DataFrame(rows)
        st.dataframe(out_df)

if __name__ == "__main__":
    main()

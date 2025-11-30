import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from database import get_all_stocks, get_stock_daily

st.set_page_config(layout="wide", page_title="A-share Dashboard")

def main():
    st.title("A-share Market Dashboard")
    
    # Sidebar: Stock Selection
    st.sidebar.header("Selection")
    
    # Load stocks
    try:
        stocks_df = get_all_stocks()
    except Exception as e:
        st.error(f"Error loading stocks: {e}")
        return

    if stocks_df.empty:
        st.warning("No stock data found. Please run the data fetcher first.")
        return
        
    # Search box
    # Create a display label: "000001.SZ - 平安银行"
    stocks_df['display'] = stocks_df['symbol'] + " - " + stocks_df['name']
    
    selected_option = st.sidebar.selectbox(
        "Select Stock",
        options=stocks_df['display'],
        index=0
    )
    
    # Date Range Selection (Global)
    st.sidebar.subheader("Date Range")
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    
    date_range = st.sidebar.date_input(
        "Select Range",
        value=(start_date, end_date)
    )
    
    # Sidebar: Data Management
    st.sidebar.markdown("---")
    st.sidebar.header("Data Management")
    
    # Initialize session state for updating
    if 'updating' not in st.session_state:
        st.session_state.updating = False
        
    if st.sidebar.button("Start/Resume Update"):
        st.session_state.updating = True
        
    if st.sidebar.button("Stop Update"):
        st.session_state.updating = False
        
    if st.session_state.updating:
        from fetcher import update_all
        
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        def update_progress(progress, message):
            progress_bar.progress(progress)
            status_text.text(message)
            
        def should_stop():
            # This function is called inside the loop.
            # However, Streamlit script execution model means if we click "Stop", 
            # the script RERUNS from top.
            # So this function might not be reachable or useful in the standard way.
            # BUT, since we are running `update_all` in the MAIN thread of the script,
            # the UI is blocked. We can't click "Stop" effectively unless we run in a separate thread?
            # Or rely on Streamlit's behavior that interaction KILLS the script.
            # If interaction kills the script, `update_all` stops automatically.
            # Then the script reruns. We check `st.session_state.updating`.
            # If we clicked "Stop", we should set `updating` to False.
            # But the button click event happens in the NEW run.
            # So, effectively, clicking ANY button stops the current run.
            # If we click "Stop", the new run sets updating=False, so it doesn't restart.
            # If we click "Start", the new run sets updating=True, so it starts.
            # So we don't need `should_stop` callback really, unless we want graceful shutdown.
            return False 

        with st.spinner("Updating data... (Click 'Stop' to pause)"):
            try:
                # Run update
                update_all(progress_callback=update_progress)
                st.sidebar.success("Data updated successfully!")
                st.session_state.updating = False # Reset
                status_text.empty()
                progress_bar.empty()
                st.cache_data.clear()
            except Exception as e:
                st.sidebar.error(f"Update failed: {e}")
                st.session_state.updating = False

    st.sidebar.markdown("---")
    
    # Main Content
    tab1, tab2, tab3 = st.tabs(["Dashboard", "Strategy Scanner", "Database Viewer"])
    
    with tab1:
        if selected_option:
            symbol = selected_option.split(" - ")[0]
            stock_info = stocks_df[stocks_df['symbol'] == symbol].iloc[0]
            
            # Display Stock Info
            st.header(f"{stock_info['name']} ({symbol})")
            col1, col2, col3 = st.columns(3)
            col1.metric("Sector", stock_info['sector'])
            col1.metric("Listing Date", stock_info['listing_date'])
            
            # Fetch Data
            # Use sidebar date range
            if len(date_range) == 2:
                start_str = date_range[0].strftime("%Y%m%d")
                end_str = date_range[1].strftime("%Y%m%d")
                
                df = get_stock_daily(symbol, start_str, end_str)
                
                if not df.empty:
                    # Calculate KDJ
                    low_list = df['low'].rolling(window=9, min_periods=1).min()
                    high_list = df['high'].rolling(window=9, min_periods=1).max()
                    
                    # Avoid division by zero
                    range_high_low = high_list - low_list
                    range_high_low.replace(0, 1e-9, inplace=True) # Small epsilon
                    
                    rsv = (df['close'] - low_list) / range_high_low * 100
                    
                    df['k'] = rsv.ewm(com=2, adjust=False).mean()
                    df['d'] = df['k'].ewm(com=2, adjust=False).mean()
                    df['j'] = 3 * df['k'] - 2 * df['d']
                    
                    # Format date for better display
                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                    
                    # Plotting
                    fig = go.Figure()
                    
                    from plotly.subplots import make_subplots
                    # 3 rows: Price, Volume, KDJ
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                        vertical_spacing=0.03, subplot_titles=('OHLC', 'Volume', 'KDJ'),
                                        row_width=[0.2, 0.2, 0.6]) 
                    
                    # Row 1: Candlestick
                    fig.add_trace(go.Candlestick(
                        x=df['date'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name='Price'
                    ), row=1, col=1)
                    
                    # Row 2: Volume
                    colors = ['red' if row['close'] >= row['open'] else 'green' for index, row in df.iterrows()]
                    
                    fig.add_trace(go.Bar(
                        x=df['date'], 
                        y=df['volume'], 
                        name='Volume',
                        marker_color=colors
                    ), row=2, col=1)
                    
                    # Row 3: KDJ
                    fig.add_trace(go.Scatter(x=df['date'], y=df['k'], name='K', line=dict(color='orange', width=1)), row=3, col=1)
                    fig.add_trace(go.Scatter(x=df['date'], y=df['d'], name='D', line=dict(color='blue', width=1)), row=3, col=1)
                    fig.add_trace(go.Scatter(x=df['date'], y=df['j'], name='J', line=dict(color='purple', width=1)), row=3, col=1)
                    
                    fig.update_layout(
                        title=f"{stock_info['name']} Daily Chart",
                        yaxis_title='Price',
                        xaxis_rangeslider_visible=False,
                        height=900,
                        dragmode='pan'
                    )
                    
                    fig.update_xaxes(type='category')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("View Daily Data"):
                        st.dataframe(df)
                else:
                    st.info("No data available for the selected range.")
    
    with tab2:
        st.header("Z-ge Zhanfa Scanner")
        st.markdown("""
        **Strategy Rules:**
        1. **KDJ J-value < 13**: Indicates oversold condition (lower is better).
        2. **Volume Power > 1**: Average volume on Up days (Red) > Average volume on Down days (Green).
        """)
        
        if st.button("Start Scan"):
            from strategy import scan_stocks
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def scan_progress(progress, message):
                progress_bar.progress(progress)
                status_text.text(message)
                
            with st.spinner("Scanning stocks... This may take a few minutes."):
                try:
                    results_df = scan_stocks(progress_callback=scan_progress)
                    status_text.empty()
                    progress_bar.empty()
                    
                    if not results_df.empty:
                        st.success(f"Found {len(results_df)} stocks matching criteria!")
                        
                        # Sort by J value ascending
                        results_df = results_df.sort_values('j_value')
                        
                        st.dataframe(results_df.style.format({
                            'j_value': '{:.2f}',
                            'vol_power': '{:.2f}',
                            'close': '{:.2f}'
                        }))
                    else:
                        st.warning("No stocks found matching the criteria.")
                except Exception as e:
                    st.error(f"Scan failed: {e}")
    
    with tab3:
        st.header("Database Viewer")
        st.subheader("Stocks Table")
        st.dataframe(stocks_df)
        
        st.subheader("Daily Prices Table (Preview)")
        if selected_option:
            symbol = selected_option.split(" - ")[0]
            st.write(f"Showing data for {symbol}")
            daily_df = get_stock_daily(symbol)
            st.dataframe(daily_df)
        else:
            st.info("Select a stock to view its daily data.")

if __name__ == "__main__":
    main()

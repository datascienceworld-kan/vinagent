from vinagent.config.logger_config import setup_logger
from vinagent.cache.redis_cache import RedisCache
from typing import Tuple, Optional
import pickle
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
import pandas as pd
import os
import matplotlib.pyplot as plt
from langchain_core.tools import tool
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# load environment variables from .env file
_ = load_dotenv()
logger = setup_logger()
redis_client = RedisCache()

def fetch_stock_data(
    symbol: str,
    start_date: str = "2020-01-01",
    end_date: str = "2025-01-01",
    output_size: str = "full"
) -> pd.DataFrame:

    logger.info(f"Start fetching data for {symbol} from {start_date} to {end_date} with output_size={output_size}")
    data_filter = None
    try:
        data, meta_data = _fetch_data(symbol, output_size=output_size)
        logger.debug(f"Raw data fetched for {symbol}: {len(data)} rows")

        # --- Filter data ---
        data_filter = data[(data.index >= start_date) & (data.index <= end_date)]
        logger.debug(f"Filtered data for {symbol}: {len(data_filter)} rows in date range")

        data_filter.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        data_filter = data_filter.reset_index()

        data_filter.rename(columns={'date': 'Date'}, inplace=True)

        data_filter['Date'] = data_filter['Date'].dt.strftime('%Y-%m-%d')

        if data_filter.empty:
            logger.warning(f"No data found for {symbol} in date range {start_date} to {end_date}")
        else: logger.info(f"Successfully fetched and filtered data for {symbol}, rows: {len(data_filter)}")
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}", exc_info=True)
        return pd.DataFrame()
        
    return data_filter


def visualize_stock_data(
    symbol: str,
    start_date: str = "2020-01-01",
    end_date: str = "2025-01-01",
    output_size: str = "full",
    show_static: bool = True,
    show_interactive: bool = True
) -> Optional[go.Figure]:
    logger.info(f"Start visualizing stock data for {symbol} from {start_date} to {end_date}")
    try:
        df = fetch_stock_data(symbol, start_date, end_date, output_size)
        if df is None or df.empty:
            logger.warning(f"No data available to visualize for {symbol}")
            return
        
        logger.debug(f"Fetched dataframe for visualization: {len(df)} rows")

        # --- Ensure datetime and required columns ---
        df["Date"] = pd.to_datetime(df["Date"])
        required_columns = {"Date", "Open", "High", "Low", "Close", "Volume"}
        if not required_columns.issubset(df.columns):
            logger.error(f"Missing required columns: {required_columns - set(df.columns)}")
            return None
        
        # --- Plot static with Matplotlib ---
        if show_static:
            _plot_static_chart(df, symbol)

        # --- Plot interactive with Plotly ---
        fig = None
        if show_interactive:
            fig = _plot_interactive_chart(df, symbol)

        logger.info(f"Visualization completed for {symbol}")
        return fig
    
    except Exception as e:
        logger.error(f"Error during visualization for {symbol}: {e}", exc_info=True)
        return None
        

@tool
def plot_returns(
    symbol: str,
    start_date: str = "2020-01-01",
    end_date: str = "2025-01-01",
    output_size: str = "full"
) -> Optional[go.Figure]:
    """
    Visualize cumulative returns of the stock.
    """
    logger.info(f"Start plotting cumulative returns for {symbol} from {start_date} to {end_date}")
    try:
        df = fetch_stock_data(symbol, start_date, end_date, output_size)
        if df is None or df.empty:
            logger.warning(f"No data available to calculate returns for {symbol}")
            return
        
        logger.debug(f"Fetched {len(df)} rows for {symbol} returns calculation")

        # Calculate returns
        df["Daily_Return"] = df["Close"].pct_change()
        df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod() - 1
        logger.debug(f"Calculated daily and cumulative returns for {symbol}")

        # Plot with Plotly
        logger.info(f"Generating cumulative return chart for {symbol}")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Cumulative_Return"] * 100,
                mode="lines",
                name="Cumulative Return",
                line=dict(color="green"),
            )
        )

        fig.update_layout(
            title=f"{symbol} Cumulative Returns",
            xaxis_title="Date",
            yaxis_title="Return (%)",
            template="plotly_white",
            height=500,
        )

        fig.show()
        logger.info(f"Return chart successfully rendered for {symbol}")
        return fig
    
    except Exception as e:
        logger.error(f"Error generating return chart for {symbol}: {e}", exc_info=True)
        return None

     
def _fetch_data(
    symbol: str,
    output_size: str = "full",
    redis_client: Optional[RedisCache] = None
) -> Tuple[pd.DataFrame, dict[str, str]]:
    """
    Fetch daily stock data from Alpha Vantage via TimeSeries API.

    Returns:
        Tuple of (dataframe, metadata)
    """

    # Use provided or default Redis cache
    if redis_client is None:
        redis_client = RedisCache()

    redis_key = f"av:{symbol}:{output_size}"

    logger.info(f"Fetching data from Alpha Vantage for {symbol} with output_size={output_size}")
    try:
        # Try load from cache
        cached = redis_client.get(redis_key)

        if cached:
            logger.info(f"Loaded '{symbol}' from Redis cache")
            cached_obj = pickle.loads(cached)
            return cached_obj['data'], cached_obj['meta_data']
        
        # Cache miss → fetch from API
        logger.info(f"Fetching '{symbol}' from Alpha Vantage...")

        ts = TimeSeries(key=os.environ.get("ALPHA_VANTAGE_API_KEY"), output_format='pandas')
        data, meta_data = ts.get_daily(symbol=symbol, outputsize=output_size)

        logger.debug(f"Received {len(data)} rows for {symbol} from Alpha Vantage")

        # Format the index
        data.index = pd.to_datetime(data.index)
        data = data.sort_index(ascending=False)

        serialized = pickle.dumps({'data': data, 'meta_data': meta_data})
        redis_client.set(redis_key, serialized)

        logger.info(f"Successfully fetched and processed data for {symbol}")
        return data, meta_data
    
    except Exception as e:
        logger.error(f"Error fetching data from Alpha Vantage for {symbol}: {e}", exc_info=True)
        return pd.DataFrame(), {}
    

def _plot_static_chart(df: pd.DataFrame, symbol: str):
    logger.info(f"Generating static chart with matplotlib for {symbol}")
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(df["Date"], df["Close"], label="Close Price", color="blue")
    plt.title(f"{symbol} Stock Price and Volume")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.bar(df["Date"], df["Volume"], color="gray")
    plt.ylabel("Volume")
    plt.xlabel("Date")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def _plot_interactive_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    logger.info(f"Generating interactive candlestick chart with plotly for {symbol}")

    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Candlestick", "Volume"),
        row_heights=[0.7, 0.3],
    )

    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MA20"],
            line=dict(color="purple", width=1),
            name="20-day MA",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["EMA20"],
            line=dict(color="orange", width=1),
            name="20-day EMA",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(x=df["Date"], y=df["Volume"], name="Volume", marker_color="gray"),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"{symbol} Stock Price Analysis",
        yaxis_title="Price ($)",
        height=800,
        showlegend=True,
        template="plotly_white",
    )
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    fig.show()
    return fig
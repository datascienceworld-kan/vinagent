from datetime import datetime
from vinagent.register import primary_function
import numpy as np
import requests
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import datetime
from typing import Union
import os
from dotenv import load_dotenv
from tavily import TavilyClient
from dataclasses import dataclass
from typing import Union, Any
from plotly.subplots import make_subplots
from vnstock import Vnstock

_ = load_dotenv()

def fetch_stock_data_vn(symbol: str, start_date: str = "2020-01-01", end_date: str = "2025-01-01", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical stock data from Vnstock.

    Args:
        symbol (str): The stock symbol (e.g., 'FPT' for FPT Corporation).
        start_date (str): Start date for historical data (YYYY-MM-DD).
        end_date (str): End date for historical data (YYYY-MM-DD).
        interval (str): Data interval ('1d', '1wk', '1mo', etc.).

    Returns:
        pd.DataFrame: DataFrame containing historical stock prices.
    """
    try:
        stock = Vnstock().stock(symbol=symbol, source='VCI')
    
        # 2. Fetch dữ liệu lịch sử giá (quote history)  
        df = stock.quote.history(   
            start=start_date, 
            end=end_date, 
            interval=interval # Khung thời gian: '1D' (1 ngày), '1M' (1 tháng), v.v.
        )
    
        # Hiển thị 5 dòng đầu tiên
        return df
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        print(error_msg)
        return error_msg

def visualize_stock_data_vn(
    symbol: str,
    start_date: str = "2020-01-01",
    end_date: str = "2025-01-01",
    interval: str = "1d",
) -> None:
    """
    Visualize stock data with multiple chart types.

    Args:
        symbol (str): Stock symbol (e.g., 'AAPL')
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD). It must be greater than start_date.
        interval (str): Data interval ('1d', '1wk', '1mo')
    """
    # Fetch the data
    df = fetch_stock_data_vn(symbol, start_date, end_date, interval)
    if df is None:
        return

    # Reset index for easier plotting
    df = df.reset_index()

    # 1. Matplotlib - Price and Volume Plot
    plt.figure(figsize=(12, 8))

    # Price subplot
    plt.subplot(2, 1, 1)
    plt.plot(df["time"], df["close"], label="Close Price", color="blue")
    plt.title(f"{symbol} Stock Price and Volume")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)

    # Volume subplot
    plt.subplot(2, 1, 2)
    plt.bar(df["time"], df["volume"], color="gray")
    plt.ylabel("Volume")
    plt.xlabel("Date")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 2. Plotly - Interactive Candlestick Chart with Moving Average
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Candlestick", "Volume"),
        row_heights=[0.7, 0.3],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )

    # 20-day Moving Average
    df["MA20"] = df["close"].rolling(window=20).mean()
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["MA20"],
            line=dict(color="purple", width=1),
            name="20-day MA",
        ),
        row=1,
        col=1,
    )

    # Volume
    fig.add_trace(
        go.Bar(x=df["time"], y=df["volume"], name="Volume", marker_color="gray"),
        row=2,
        col=1,
    )

    # Update layout
    fig.update_layout(
        title=f"{symbol} Stock Price Analysis",
        yaxis_title="Price ($)",
        height=800,
        showlegend=True,
        template="plotly_white",
    )

    # Update axes
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    fig.show()
    return fig


def plot_returns_vn(
    symbol: str,
    start_date: str = "2020-01-01",
    end_date: str = "2025-01-01",
    interval: str = "1d",
) -> None:
    """
    Visualize cumulative returns of the stock.
    """
    df = fetch_stock_data_vn(symbol, start_date, end_date, interval)
    if df is None:
        return

    # Calculate daily returns and cumulative returns
    df["Daily_Return"] = df["close"].pct_change()
    df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod() - 1

    # Plot with Plotly
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
    return fig


class WebSearchClient:
    tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

    def call_api(self, query: Union[str, dict[str, str]]):
        if isinstance(query, dict):
            query_string = "\n".join([f"{k}: {v}" for (k, v) in query.items()])
        else:
            query_string = query
        result = self.tavily_client.search(query_string, include_answer=True)
        return result["answer"]

def search_api(query: Union[str, dict[str, str]]) -> Any:
    """
    Search for an answer from a query string
    Args:
        query (dict[str, str]):  The input query to search
    Returns:
        The answer from search query
    """
    client = WebSearchClient()
    answer = client.call_api(query)
    return answer

def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fetch_btc_market_data() -> str:
    """
    Fetch comprehensive Bitcoin (BTC) market data including price, changes (daily/weekly/monthly), and moving averages from the CoinDesk API.
    Returns:
        str: A comprehensive summary of BTC market metrics.
    """
    try:
        response = requests.get('https://data-api.coindesk.com/index/cc/v1/latest/tick',
                params={"market":"cadli","instruments":"BTC-USD","apply_mapping":"true"},
                headers={"Content-type":"application/json; charset=UTF-8"}
        )
        data = response.json()['Data']['BTC-USD']
        
        summary = (
            f"--- BTC-USD Market Data ---\n"
            f"Current Price: ${data['VALUE']:.2f} USD\n"
            f"Day Change: {data['CURRENT_DAY_CHANGE']:.2f} ({data['CURRENT_DAY_CHANGE_PERCENTAGE']:.2f}%)\n"
            f"Week Change: {data['CURRENT_WEEK_CHANGE']:.2f} ({data['CURRENT_WEEK_CHANGE_PERCENTAGE']:.2f}%)\n"
            f"Month Change: {data['CURRENT_MONTH_CHANGE']:.2f} ({data['CURRENT_MONTH_CHANGE_PERCENTAGE']:.2f}%)\n"
            f"24h Range: ${data['MOVING_24_HOUR_LOW']:.2f} - ${data['MOVING_24_HOUR_HIGH']:.2f}\n"
            f"Year Change: {data['CURRENT_YEAR_CHANGE_PERCENTAGE']:.2f}%\n"
            f"Volatility Context: Day High ${data['CURRENT_DAY_HIGH']:.2f} / Low ${data['CURRENT_DAY_LOW']:.2f}"
        )
        return summary
    except Exception as e:
        return f"Error fetching BTC price: {e}"


@primary_function
def calculate_black_litterman(symbols: list, priors: list, views: list, tau: float = 0.05) -> dict:
    """
    Perform Black-Litterman calculation to get posterior returns.
    
    Args:
        symbols (list): A list of stock ticker symbols (e.g., ['FPT', 'CMG', 'VGI']).
        priors (list): A list of prior expected returns (floats) for each symbol.
        views (list): A list of subjective market views (dictionaries). Leave empty [] if none.
        tau (float): Scalar indicating the uncertainty of the prior.
    """
    try:
        import numpy as np
        n = len(symbols)
        sigma = np.eye(n) * 0.05
        pi = np.array(priors).reshape(-1, 1)
        k = len(views)
        if k == 0: return {symbols[i]: float(priors[i]) for i in range(n)}
        P, Q, Omega = np.zeros((k, n)), np.zeros((k, 1)), np.zeros((k, k))
        for i, v in enumerate(views):
            for s, w in zip(v['symbols'], v['weights']):
                if s in symbols: P[i, symbols.index(s)] = w
            Q[i, 0], Omega[i, i] = v['return'], (1.001 - v['confidence']) * 0.1
        ts_inv, o_inv = np.linalg.inv(tau * sigma), np.linalg.inv(Omega)
        res = np.linalg.inv(ts_inv + P.T @ o_inv @ P) @ (ts_inv @ pi + P.T @ o_inv @ Q)
        return {symbols[i]: round(float(res[i, 0]), 4) for i in range(n)}
    except Exception as e: return {"error": str(e)}

@primary_function
def optimize_portfolio(symbols: list, expected_returns: dict, risk_aversion: float = 2.5) -> dict:
    """
    Perform Mean-Variance Optimization to generate target weights.
    
    Args:
        symbols (list): A list of stock ticker symbols.
        expected_returns (dict): Dictionary mapping symbols to their expected returns.
        risk_aversion (float): The risk aversion coefficient.
    """
    try:
        import numpy as np
        n = len(symbols)
        sigma = np.eye(n) * 0.05
        mu = np.array([expected_returns[s] for s in symbols]).reshape(-1, 1)
        w = np.linalg.inv(risk_aversion * sigma) @ mu
        wn = np.maximum(w, 0)
        if wn.sum() > 0: wn = wn / wn.sum()
        return {symbols[i]: round(float(wn[i, 0]), 4) for i in range(n)}
    except Exception as e: return {"error": str(e)}

@primary_function
def calculate_equilibrium_returns(symbols: list, risk_free_rate: float = 0.03, market_risk_premium: float = 0.05) -> dict:
    """
    Calculate CAPM-based equilibrium returns (priors) for Black-Litterman.
    
    Args:
        symbols (list): A list of stock ticker symbols (e.g., ['FPT', 'CMG', 'VGI']).
        risk_free_rate (float): The risk-free rate.
        market_risk_premium (float): The expected market premium.
    """
    try:
        results = {}
        for s in symbols:
            # Updated to include Vietnamese Tech/Telecom stocks
            betas = {"FPT": 1.1, "CMG": 1.3, "VGI": 1.2, "ELC": 1.4, "ITD": 1.1}
            beta = betas.get(s, 1.0)
            expected_return = risk_free_rate + beta * market_risk_premium
            results[s] = round(expected_return, 4)
        return results
    except Exception as e:
        return {"error": str(e)}

def backtest_alpha_strategy(symbol: str, strategy_type: str, start_date: str = "2020-01-01", end_date: str = "2025-01-01", **kwargs) -> dict:
    """
    Backtest a specific alpha strategy.
    Supported strategy_type: 'momentum', 'mean_reversion'.
    """
    try:
        df = fetch_stock_data_vn(symbol, start_date, end_date)
        if df is None or df.empty:
            return {"error": f"No data found for {symbol}"}
        
        df['returns'] = df['close'].pct_change()
        
        if strategy_type == 'momentum':
            window = kwargs.get('window', 20)
            df['signal'] = np.where(df['close'] > df['close'].shift(window), 1, -1)
        elif strategy_type == 'mean_reversion':
            window = kwargs.get('window', 20)
            ma = df['close'].rolling(window).mean()
            std = df['close'].rolling(window).std()
            df['signal'] = np.where(df['close'] < ma - std, 1, np.where(df['close'] > ma + std, -1, 0))
        else:
            return {"error": f"Unsupported strategy type: {strategy_type}"}
            
        df['strategy_returns'] = df['signal'].shift(1) * df['returns']
        df = df.dropna()
        
        cum_returns = (1 + df['strategy_returns']).cumprod()
        sharpe = (df['strategy_returns'].mean() / df['strategy_returns'].std()) * np.sqrt(252) if df['strategy_returns'].std() != 0 else 0
        
        return {
            "strategy": strategy_type,
            "total_return": round(cum_returns.iloc[-1] - 1, 4),
            "annualized_return": round((cum_returns.iloc[-1]**(252/len(df))) - 1, 4) if len(df) > 0 else 0,
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown": round(((cum_returns / cum_returns.cummax()) - 1).min(), 4)
        }
    except Exception as e:
        return {"error": str(e)}

def calculate_equilibrium_returns(symbols: list, risk_free_rate: float = 0.03, market_risk_premium: float = 0.05) -> dict:
    """
    Calculate CAPM-based equilibrium returns (priors) for Black-Litterman.
    
    Args:
        symbols (list): A list of stock ticker symbols (e.g., ['SPY', 'AGG']).
        risk_free_rate (float): The risk-free rate.
        market_risk_premium (float): The expected market premium.
    """
    try:
        results = {}
        for s in symbols:
            # Simplified mock betas for demonstration if not fetching real market beta
            # In a real tool, we might fetch index vs stock covariance
            betas = {"FPT": 1.1, "HPG": 1.3, "VIB": 1.2, "VCB": 1.1, "VHM": 1.2}
            beta = betas.get(s, 1.0)
            expected_return = risk_free_rate + beta * market_risk_premium
            results[s] = round(expected_return, 4)
        return results
    except Exception as e:
        return {"error": str(e)}

def main():
    symbol = "FPT"
    start_date = "2022-01-01"
    end_date = "2022-12-31"
    interval = "1d"
    df = fetch_stock_data_vn(symbol, start_date, end_date, interval)
    print(df)
    print("Visualizing stock data...")
    visualize_stock_data_vn(symbol, start_date, end_date, interval)
    print("Plotting returns...")
    plot_returns_vn(symbol, start_date, end_date, interval)

if __name__ == "__main__":
    main()

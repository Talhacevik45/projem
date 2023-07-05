#!/usr/bin/env python
# coding: utf-8

# In[1]:


filepath = "bitcoin.csv"
symbol = "BTCUSDT"
start = "2017-08-17"
end = "2021-10-07"
tc = -0.00085
sma_s = 15
sma_m = 50
sma_l = 200


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


data = pd.read_csv("bitcoin.csv", parse_dates = ["Date"], index_col = "Date")
data


# In[4]:


data = data[["Close"]].copy()


# In[5]:


data


# In[6]:


data["returns"] = np.log(data.Close.div(data.Close.shift(1)))


# In[7]:


data


# In[8]:


SMA_S = 15
SMA_M = 50
SMA_L = 200


# In[9]:


data["SMA_S"] = data.Close.rolling(window = SMA_S).mean()
data["SMA_M"] = data.Close.rolling(window = SMA_M).mean()
data["SMA_L"] = data.Close.rolling(window = SMA_L).mean()


# In[10]:


data


# In[11]:


data.dropna(inplace = True)


# In[12]:


smas = ["SMA_S", "SMA_M", "SMA_L"]
smas


# In[13]:


data["position"] = 0 # Trading position -> Neutral for all bars
data


# In[14]:


cond1 = (data.SMA_S > data.SMA_M) & (data.SMA_M > data.SMA_L) # long position
cond1


# In[15]:


cond2 = (data.SMA_S < data.SMA_M) & (data.SMA_M < data.SMA_L) # short position
cond2


# In[16]:


data.loc[cond1, "position"] = 1
data.loc[cond2, "position"] = -1


# In[17]:


data.loc[:, smas + ["position"]].plot(figsize = (12, 8), secondary_y = "position")
plt.show()


# In[18]:


import pandas as pd
import numpy as np
from itertools import product

from binance.client import Client
import pandas as pd
import ta

class Trader:
    def __init__(self, client):
        self.client = client
        self.df = None
        self.short_sma_column = "Short_SMA"
        self.long_sma_column = "Long_SMA"

    def get_data(self, symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1DAY, lookback=100):
        klines = self.client.get_historical_klines(symbol, interval, f"{lookback} day ago UTC")
        data = pd.DataFrame(klines, columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

        data['Open'] = pd.to_numeric(data['Open'])
        data['High'] = pd.to_numeric(data['High'])
        data['Low'] = pd.to_numeric(data['Low'])
        data['Close'] = pd.to_numeric(data['Close'])
        data['Volume'] = pd.to_numeric(data['Volume'])

        self.df = data

    def calculate_SMA(self, window):
        self.df[f'SMA_{window}'] = self.df['Close'].rolling(window).mean()

    def calculate_RSI(self, window):
        delta = self.df['Close'].diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        average_gain = up.rolling(window=window).mean()
        average_loss = abs(down.rolling(window=window).mean())

        rs = average_gain / average_loss
        rsi = 100 - (100 / (1 + rs))
        
        self.df[f'RSI_{window}'] = rsi

    def trade(self, short_window, long_window):
        self.short_sma_column = f'SMA_{short_window}'
        self.long_sma_column = f'SMA_{long_window}'

        self.calculate_SMA(short_window)
        self.calculate_SMA(long_window)
        self.calculate_RSI(14)

        buys = []
        sells = []

        for i in range(1, len(self.df)):
            if self.df[self.short_sma_column].iloc[i] > self.df[self.long_sma_column].iloc[i] and self.df[self.short_sma_column].iloc[i - 1] < self.df[self.long_sma_column].iloc[i - 1]:
                # Buy signal
                buys.append((self.df['timestamp'].iloc[i], self.df['Close'].iloc[i]))
            elif self.df[self.short_sma_column].iloc[i] < self.df[self.long_sma_column].iloc[i] and self.df[self.short_sma_column].iloc[i - 1] > self.df[self.long_sma_column].iloc[i - 1]:
                # Sell signal
                sells.append((self.df['timestamp'].iloc[i], self.df['Close'].iloc[i]))

        return buys, sells

class Long_Short_Backtester():
    def __init__(self, filepath, symbol, start, end, tc):
        
        self.filepath = filepath
        self.symbol = symbol
        self.start = start
        self.end = end
        self.tc = tc
        self.results = None
        self.get_data()
        self.tp_year = (self.data.Close.count() / ((self.data.index[-1] - self.data.index[0]).days / 365.25))
        
    def __repr__(self):
        return "Long_Short_Backtester(symbol = {}, start = {}, end = {})".format(self.symbol, self.start, self.end)
   
    @staticmethod
    def correct_date_format(date_str):
        if date_str.endswith(".000"):
            return date_str[:-4]  # remove the last 4 characters ".000"
        return date_str
    
    def get_data(self):
        raw = pd.read_csv(self.filepath, index_col="Date")
    
        # Correct the date format
        raw.index = raw.index.map(self.correct_date_format)
    
        # Try different datetime formats
        try:       
            raw.index = pd.to_datetime(raw.index, format='%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            try:
                raw.index = pd.to_datetime(raw.index, format='%Y-%m-%d %H:%M:%S')
            except ValueError:
                raw.index = pd.to_datetime(raw.index, infer_datetime_format=True)

    
        raw = raw.loc[self.start:self.end].copy()
        raw["returns"] = np.log(raw.Close / raw.Close.shift(1))
        self.data = raw
        
    def test_strategy(self, smas):
        self.SMA_S = smas[0]
        self.SMA_M = smas[1]
        self.SMA_L = smas[2]
        
        
        self.prepare_data(smas = smas)
        self.run_backtest()
        
        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        
        self.print_performance()
    
    def prepare_data(self, smas):
        ########################## Strategy-Specific #############################
        
        data = self.data[["Close", "returns"]].copy()
        data["SMA_S"] = data.Close.rolling(window = smas[0]).mean()
        data["SMA_M"] = data.Close.rolling(window = smas[1]).mean()
        data["SMA_L"] = data.Close.rolling(window = smas[2]).mean()
        
        data.dropna(inplace = True)
                
        cond1 = (data.SMA_S > data.SMA_M) & (data.SMA_M > data.SMA_L)
        cond2 = (data.SMA_S < data.SMA_M) & (data.SMA_M < data.SMA_L)
        
        data["position"] = 0
        data.loc[cond1, "position"] = 1
        data.loc[cond2, "position"] = -1

        ##########################################################################
        
        self.results = data
    
    def run_backtest(self):
        ''' Runs the strategy backtest.
        '''
        
        data = self.results.copy()
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data["trades"] = data.position.diff().fillna(0).abs()
        data.strategy = data.strategy + data.trades * self.tc
        
        self.results = data
    
    def plot_results(self):
        '''  Plots the cumulative performance of the trading strategy compared to buy-and-hold.
        '''
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            title = "{} | TC = {}".format(self.symbol, self.tc)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
            
    def optimize_strategy(self, SMA_S_range, SMA_M_range, SMA_L_range, metric="Multiple"):

        self.metric = metric
        
        if metric == "Multiple":
            performance_function = self.calculate_multiple
        elif metric == "Sharpe":
            performance_function = self.calculate_sharpe
        
        SMA_S_range = range(*SMA_S_range)
        SMA_M_range = range(*SMA_M_range)
        SMA_L_range = range(*SMA_L_range)
        
        combinations = list(product(SMA_S_range, SMA_M_range, SMA_L_range))
         
        performance = []
        for comb in combinations:
            self.prepare_data(smas = comb)
            self.run_backtest()
            performance.append(performance_function(self.results.strategy))
    
        self.results_overview = pd.DataFrame(data=np.array(combinations), columns=["SMA_S", "SMA_M", "SMA_L"])
        self.results_overview["performance"] = performance
        self.find_best_strategy()
        
        
    def find_best_strategy(self):
 
        best = self.results_overview.nlargest(1, "performance")
        SMA_S = best.SMA_S.iloc[0]
        SMA_M = best.SMA_M.iloc[0]
        SMA_L = best.SMA_L.iloc[0]
        perf = best.performance.iloc[0]
        print("SMA_S: {} | SMA_M: {} | SMA_L : {} | {}: {}".format(SMA_S, SMA_M, SMA_L, self.metric, round(perf, 5)))  
        self.test_strategy(smas=(SMA_S, SMA_M, SMA_L))
            
    ############################## Performance ######################################
    
    def print_performance(self):

        data = self.results.copy()
        strategy_multiple = round(self.calculate_multiple(data.strategy), 6)
        bh_multiple = round(self.calculate_multiple(data.returns), 6)
        outperf = round(strategy_multiple - bh_multiple, 6)
        cagr = round(self.calculate_cagr(data.strategy), 6)
        ann_mean = round(self.calculate_annualized_mean(data.strategy), 6)
        ann_std = round(self.calculate_annualized_std(data.strategy), 6)
        sharpe = round(self.calculate_sharpe(data.strategy), 6)
       
        print(100 * "=")
        print("TRIPLE SMA STRATEGY | INSTRUMENT = {} | SMAs = {}".format(self.symbol, [self.SMA_S, self.SMA_M, self.SMA_L]))
        print(100 * "-")
        print("PERFORMANCE MEASURES:")
        print("\n")
        print("Multiple (Strategy):         {}".format(strategy_multiple))
        print("Multiple (Buy-and-Hold):     {}".format(bh_multiple))
        print(38 * "-")
        print("Out-/Underperformance:       {}".format(outperf))
        print("\n")
        print("CAGR:                        {}".format(cagr))
        print("Annualized Mean:             {}".format(ann_mean))
        print("Annualized Std:              {}".format(ann_std))
        print("Sharpe Ratio:                {}".format(sharpe))
        
        print(100 * "=")
        
    def calculate_multiple(self, series):
        return np.exp(series.sum())
    
    def calculate_cagr(self, series):
        return np.exp(series.sum()) ** (1 / ((series.index[-1] - series.index[0]).days / 365.25)) - 1
    
    def calculate_annualized_mean(self, series):
        return series.mean() * self.tp_year
    
    def calculate_annualized_std(self, series):
        return series.std() * np.sqrt(self.tp_year)
    
    def calculate_sharpe(self, series):
        if series.std() == 0:
            return np.nan
        else:
            return self.calculate_cagr(series) / self.calculate_annualized_std(series)


# In[19]:


filepath = "bitcoin.csv"
symbol = "BTCUSDT"
start = "2017-08-17"
end = "2021-10-07"
tc = -0.00085
sma_s = 15 
sma_m = 50 
sma_l = 200 


# In[20]:


tester = Long_Short_Backtester(filepath = "bitcoin.csv", symbol = symbol,
                              start = start, end = end, tc = tc)


# In[21]:


tester.data


# In[22]:


filepath = "bitcoin.csv"


# In[23]:


tester.test_strategy(smas = (sma_s, sma_m, sma_l))


# In[24]:


tester.plot_results()


# In[25]:


tester.results


# In[26]:


tester.results.trades.value_counts()


# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import warnings
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")


# In[28]:


tester.optimize_strategy(SMA_S_range=(20, 50, 5),
                         SMA_M_range=(50, 100, 5),
                         SMA_L_range=(100, 200, 10),
                         metric="Multiple")


# In[29]:


tester.plot_results()


# In[30]:


tester.results_overview


# In[31]:


from binance.client import Client
from binance import ThreadedWebsocketManager
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time


# In[38]:


api_key = "yCqPvk02Cj7sp2wuAm1GCKzu9Hv1aqlKsnQT1WamSbIBpTweKX8bBBX69sGfZlBt"
secret_key = "AgUGYXGJnpT2ht2ak0C6F91JmYtYXHWDJItc9tV9qgGlteDAjNbndxyMPSHfa9Kz"


# In[39]:


client = Client(api_key = api_key, api_secret = secret_key, tld = "com", testnet = True)


# In[40]:


class LongShortTrader():
    
    def __init__(self, symbol, bar_length, sma_s, sma_m, sma_l, units, position = 0):
        
        self.symbol = symbol
        self.bar_length = bar_length
        self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
        self.units = units
        self.position = position
        self.trades = 0 
        self.trade_values = []
        
        #*****************add strategy-specific attributes here******************
        self.SMA_S = sma_s
        self.SMA_M = sma_m
        self.SMA_L = sma_l
        #************************************************************************
    
    def start_trading(self, historical_days):
        self.bar_length = "15m"  # 15 dakikalık mumlar için
        self.twm = ThreadedWebsocketManager()
        self.twm.start()
        
        if self.bar_length in self.available_intervals:
            self.get_most_recent(symbol = self.symbol, interval = self.bar_length,
                                 days = historical_days)
            self.twm.start_kline_socket(callback = self.stream_candles,
                                        symbol = self.symbol, interval = self.bar_length)
        # "else" to be added later in the course 
    
    def get_most_recent(self, symbol, interval, days):
    
        now = datetime.utcnow()
        past = str(now - timedelta(days = days))
    
        bars = client.get_historical_klines(symbol = symbol, interval = interval,
                                            start_str = past, end_str = None, limit = 1000)
        df = pd.DataFrame(bars)
        df["Date"] = pd.to_datetime(df.iloc[:,0], unit = "ms")
        df.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                      "Clos Time", "Quote Asset Volume", "Number of Trades",
                      "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore", "Date"]
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        df.set_index("Date", inplace = True)
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors = "coerce")
        df["Complete"] = [True for row in range(len(df)-1)] + [False]
        
        self.data = df
    
    def stream_candles(self, msg):
        
        # extract the required items from msg
        event_time = pd.to_datetime(msg["E"], unit = "ms")
        start_time = pd.to_datetime(msg["k"]["t"], unit = "ms")
        first   = float(msg["k"]["o"])
        high    = float(msg["k"]["h"])
        low     = float(msg["k"]["l"])
        close   = float(msg["k"]["c"])
        volume  = float(msg["k"]["v"])
        complete=       msg["k"]["x"]
        # stop trading session
        if self.position != 0 and len(self.prepared_data) > 0 and len(self.prepared_data["position"]) > 0:
            self.twm.stop()
            if self.position == 1:
                order = client.create_order(symbol=self.symbol, side="SELL", type="MARKET", quantity=self.units)
                self.report_trade(order, "GOING NEUTRAL AND STOP")
                self.position = 0
            elif self.position == -1:
                order = client.create_order(symbol=self.symbol, side="BUY", type="MARKET", quantity=self.units)
                self.report_trade(order, "GOING NEUTRAL AND STOP")
                self.position = 0
            else:
                print("STOP")
    
        else:
            # print out
            print(".", end="", flush=True)  # just print something to get a feedback (everything OK)
    
            # feed df (add new bar / update latest bar)
            self.data.loc[start_time] = [first, high, low, close, volume, complete]
    
            # prepare features and define strategy/trading positions whenever the latest bar is complete
            if complete == True:
                self.define_strategy()
                self.execute_trades()
    def define_strategy(self):
        
        data = self.data.copy()
        
        #******************** define your strategy here ************************
        data = data[["Close"]].copy()
        
        data["SMA_S"] = data.Close.rolling(window = self.SMA_S).mean()
        data["SMA_M"] = data.Close.rolling(window = self.SMA_M).mean()
        data["SMA_L"] = data.Close.rolling(window = self.SMA_L).mean()
        
        data.dropna(inplace = True)
                
        cond1 = (data.SMA_S > data.SMA_M) & (data.SMA_M > data.SMA_L)
        cond2 = (data.SMA_S < data.SMA_M) & (data.SMA_M < data.SMA_L)
        
        data["position"] = 0
        data.loc[cond1, "position"] = 1
        data.loc[cond2, "position"] = -1
        #***********************************************************************
        
        self.prepared_data = data.copy()
    
    def execute_trades(self):
        
        if len(self.prepared_data) > 0 and len(self.prepared_data["position"]) > 0:
            
            if self.prepared_data["position"].iloc[-1] == 1:  # if position is long -> go/stay long
                if self.position == 0:
                    order = client.create_order(symbol=self.symbol, side="BUY", type="MARKET", quantity=self.units)
                    self.report_trade(order, "GOING LONG")
                elif self.position == -1:
                    order = client.create_order(symbol=self.symbol, side="BUY", type="MARKET", quantity=self.units)
                    self.report_trade(order, "GOING NEUTRAL")
                    time.sleep(1)
                    order = client.create_order(symbol=self.symbol, side="BUY", type="MARKET", quantity=self.units)
                    self.report_trade(order, "GOING LONG")
                self.position = 1
            elif self.prepared_data["position"].iloc[-1] == 0:  # if position is neutral -> go/stay neutral
                if self.position == 1:
                    order = client.create_order(symbol=self.symbol, side="SELL", type="MARKET", quantity=self.units)
                    self.report_trade(order, "GOING NEUTRAL")
                elif self.position == -1:
                    order = client.create_order(symbol=self.symbol, side="BUY", type="MARKET", quantity=self.units)
                    self.report_trade(order, "GOING NEUTRAL")
                self.position = 0
            elif self.prepared_data["position"].iloc[-1] == -1:  # if position is short -> go/stay short
                if self.position == 0:
                    order = self.client.create_order(symbol=self.symbol, side="SELL", type="MARKET", quantity=self.units)
                    self.report_trade(order, "GOING SHORT")
                elif self.position == 1:
                    order = self.client.create_order(symbol=self.symbol, side="SELL", type="MARKET", quantity=self.units)
                    self.report_trade(order, "GOING NEUTRAL")
                    time.sleep(1)
                    order = self.client.create_order(symbol=self.symbol, side="SELL", type="MARKET", quantity=self.units)
                    self.report_trade(order, "GOING SHORT")
                self.position = -1   

   
    def report_trade(self, order, going): 
        
        side = order["side"]
        time = pd.to_datetime(order["transactTime"], unit = "ms")
        base_units = float(order["executedQty"])
        quote_units = float(order["cummulativeQuoteQty"])
        price = round(quote_units / base_units, 5)
        
        self.trades += 1
        if side == "BUY":
            self.trade_values.append(-quote_units)
        elif side == "SELL":
            self.trade_values.append(quote_units) 
        
        if self.trades % 2 == 0:
            real_profit = round(np.sum(self.trade_values[-2:]), 3) 
            self.cum_profits = round(np.sum(self.trade_values), 3)
        else: 
            real_profit = 0
            self.cum_profits = round(np.sum(self.trade_values[:-1]), 3)
        
        print(2 * "\n" + 100* "-")
        print("{} | {}".format(time, going)) 
        print("{} | Base_Units = {} | Quote_Units = {} | Price = {} ".format(time, base_units, quote_units, price))
        print("{} | Profit = {} | CumProfits = {} ".format(time, real_profit, self.cum_profits))
        print(100 * "-" + "\n")


# In[41]:


filepath = "bitcoin.csv"


# In[42]:


symbol = "BTCUSDT"
bar_length = "1m"
sma_s = 10
sma_m = 20
sma_l = 50
units = 0.001
position = 0


# In[43]:


client.get_account()


# In[44]:


trader = LongShortTrader(symbol = symbol, bar_length = bar_length, sma_s = sma_s, sma_m = sma_m, sma_l = sma_l,
                         units = units, position = position)


# In[45]:


trader.start_trading(historical_days = 1/76)


# In[ ]:


time.sleep(300)


# In[ ]:


trader.twm.stop()


# In[ ]:


trader.prepared_data


import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import talib  # TA-Lib for additional indicators
import matplotlib.pyplot as plt
import mplfinance as mpf 
import logging
import datetime
import matplotlib.dates as mdates
import os
import time
import psutil
import subprocess
import traceback

log_file_path = r"C:\Users\Gansumiya\Desktop\TradeBot\TradeVersion_4.log"
MT5_PATH = r"C:\Program Files\MetaTrader 5 IC Markets Global\terminal64.exe"

def log_message(message):
    with open(log_file_path, "a") as log_file:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] {message}\n")

def is_mt5_running():
    """Check if MetaTrader 5 is already running."""
    for process in psutil.process_iter(attrs=['name']):
        if "terminal64.exe" in process.info['name'].lower():
            return True
    return False

def restart_mt5():
    """Restart MetaTrader 5."""
    try:
        if is_mt5_running():
            log_message("Closing existing MT5 instance...")
            subprocess.run(["taskkill", "/IM", "terminal64.exe", "/F"], shell=True)
            time.sleep(30)  # Wait for MT5 to fully close

        log_message("Starting MT5...")
        subprocess.Popen([MT5_PATH], shell=True)  # Start MT5
        time.sleep(30)  # Wait for MT5 to initialize

        if is_mt5_running():
            log_message("MT5 started successfully.")
        else:
            log_message("MT5 failed to start.")

    except Exception as e:
        log_message(f"Error: {str(e)}\n{traceback.format_exc()}")

def fetch_data(symbol, timeframe , n_bars):
    # Initialize MT5
    if not mt5.initialize():
        print("MetaTrader 5 initialization failed")
        return None

    # Ensure symbol is available
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None or not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"Failed to enable symbol {symbol}, exiting.")
            mt5.shutdown()
            return None

    # Fetch historical data
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None or len(rates) == 0:
        print(f"Failed to fetch data for {symbol}. Error: {mt5.last_error()}")
        mt5.shutdown()
        return None

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Set volume column
    df['volume'] = df['real_volume'] if 'real_volume' in df.columns and df['real_volume'].sum() > 0 else df['tick_volume']
    df['symbol'] = symbol
    
    

    
    # Trend Indicators
    df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
    df['SMA_200'] = talib.SMA(df['close'], timeperiod=200)
    df['EMA_50'] = talib.EMA(df['close'], timeperiod=50)
    df['EMA_200'] = talib.EMA(df['close'], timeperiod=200)
    df['PSAR'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)  # Average Directional Index
    df['Plus_DI'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)  # Positive Directional Indicator
    df['Minus_DI'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)  # Negative Directional Indicator

    # Momentum Indicators
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    macd, signal, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['Signal'] = signal
    stoch_k, stoch_d = talib.STOCH(df['high'], df['low'], df['close'], 
                                   fastk_period=14, slowk_period=3, slowd_period=3)
    df['Stoch_K'] = stoch_k
    df['Stoch_D'] = stoch_d
    df['W%R'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
    df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
    df['MOM'] = talib.MOM(df['close'], timeperiod=14)
    df['ROC'] = talib.ROC(df['close'], timeperiod=10)  # Rate of Change
    df['RSI_2'] = talib.RSI(df['close'], timeperiod=2)  # 2-period Relative Strength Index

    # Volatility Indicators
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
    df['BB_Upper'] = upper
    df['BB_Middle'] = middle
    df['BB_Lower'] = lower
    df['Keltner_Channel_Upper'], df['Keltner_Channel_Lower'] = talib.EMA(df['close'], timeperiod=20), talib.EMA(df['close'], timeperiod=20)
    df['Keltner_Channel_Upper'] = df['Keltner_Channel_Upper'] + (df['ATR'] * 1.5)
    df['Keltner_Channel_Lower'] = df['Keltner_Channel_Lower'] - (df['ATR'] * 1.5)

    # Volume-Based Indicators
    df['OBV'] = talib.OBV(df['close'], df['tick_volume'])
    df['ADL'] = talib.AD(df['high'], df['low'], df['close'], df['tick_volume'])
    df['CMF'] = talib.ADOSC(df['high'], df['low'], df['close'], df['tick_volume'], fastperiod=3, slowperiod=10)
    df['Chaikin_Money_Flow'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)  # Chaikin Money Flow

    #Ichimoku 
    df['Tenkan_sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2  # Tenkan-sen (Conversion Line)
    df['Kijun_sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2  # Kijun-sen (Base Line)

    # Senkou Span A (Leading Span A)
    df['Senkou_span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)  # Plotted 26 periods ahead

    # Senkou Span B (Leading Span B)
    df['Senkou_span_B'] = (df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2  # Senkou Span B (52 periods)
    df['Senkou_span_B'] = df['Senkou_span_B'].shift(26)  # Plotted 26 periods ahead

    # Chikou Span (Lagging Span)
    df['Chikou_span'] = df['close'].shift(-26)  # Plotted 26 periods behind

    # Cloud (Kumo) - Shaded area between Senkou Span A and Senkou Span B
    df['Cloud_top'] = df['Senkou_span_A']
    df['Cloud_bottom'] = df['Senkou_span_B']

    # Handle NaN values to avoid errors in signal calculations
    df.fillna(0, inplace=True)

    # Adding Signals for Ichimoku Cloud
    df['Ichimoku_Bullish'] = np.where(df['close'] > df['Senkou_span_A'], 1, -1)  # Bullish signal if price is above Senkou Span A
    df['Ichimoku_Bearish'] = np.where(df['close'] < df['Senkou_span_B'], -1, 1)  # Bearish signal if price is below Senkou Span B
    df['Ichimoku_Trend'] = np.where(df['Senkou_span_A'] > df['Senkou_span_B'], 1, -1)  # Bullish if Senkou Span A is above B, otherwise bearish


    # Handle NaN values to avoid errors in signal calculations
    df.fillna(0, inplace=True)     

    df['RSI'] = df['RSI'].where(df.index >= 14, np.nan)
    
    df['Trend_SMA'] = np.where(df['SMA_50'] > df['SMA_200'], 1, -1)
    df['Trend_EMA'] = np.where(df['EMA_50'] > df['EMA_200'], 1, -1)
    df['RSI_Signal'] = np.where(df['RSI'].notna() & (df['RSI'] < 30), 1, 
                            np.where(df['RSI'] > 70, -1, 0))
    df['MACD_Signal'] = np.where(df['MACD'] > df['Signal'], 1, -1)
    df['Stoch_Signal'] = np.where(df['Stoch_K'] > df['Stoch_D'], 1, -1)
    df['W%R_Signal'] = np.where(df['W%R'].notna() & (df['W%R'] < -80), 1, 
                            np.where(df['W%R'] > -20, -1, 0))
    df['CCI_Signal'] = np.where(df['CCI'].notna() & (df['CCI'] < -100), 1, 
                            np.where(df['CCI'] > 100, -1, 0))
    df['MOM_Signal'] = np.where(df['MOM'].notna() & (df['MOM'] > 0), 1, -1)
    df['BB_Signal'] = np.where(df['close'].notna() & (df['close'] < df['BB_Lower']), 1, 
                           np.where(df['close'] > df['BB_Upper'], -1, 0))
    df['OBV_Signal'] = np.where(df['OBV'].notna() & (df['OBV'].diff() > 0), 1, -1)
    df['CMF_Signal'] = np.where(df['CMF'].notna() & (df['CMF'] > 0), 1, 
                            np.where(df['CMF'] < 0, -1, 0))
    df['ADL_Signal'] = np.where(df['ADL'].notna() & (df['ADL'].diff() > 0), 1, -1)

    df['ADX_Signal'] = np.where(df['ADX'] > 25, 1, -1)  # ADX signal: 1 when trending, -1 when not trending
    df['ROC_Signal'] = np.where(df['ROC'] > 0, 1, -1)  # ROC signal: 1 for positive momentum, -1 for negative
    df['RSI_2_Signal'] = np.where(df['RSI_2'] < 30, 1, np.where(df['RSI_2'] > 70, -1, 0))  # RSI 2 signal
    df['Keltner_Signal'] = np.where(df['close'] > df['Keltner_Channel_Upper'], -1, np.where(df['close'] < df['Keltner_Channel_Lower'], 1, 0))  # Keltner Channel signal
    df['Chaikin_Money_Flow_Signal'] = np.where(df['Chaikin_Money_Flow'] > 0, 1, -1)  # CMF signal

    price_bins = 50
    price_min = df['low'].min()
    price_max = df['high'].max()

    # Create the bins for price levels
    price_range = price_max - price_min
    price_step = price_range / price_bins
    price_levels = [price_min + i * price_step for i in range(price_bins)]

    # Create a column for volume at each price level
    volume_profile = [0] * price_bins
    for i in range(len(df)):
        close_price = df['close'][i]
        # Find the corresponding price bin
        bin_index = int((close_price - price_min) / price_step)
        if 0 <= bin_index < price_bins:
            volume_profile[bin_index] += df['tick_volume'][i]

    # Identify the Point of Control (POC)
    poc_index = volume_profile.index(max(volume_profile))
    poc_price = price_levels[poc_index]
    print("Point of Control:", poc_price)

    # Define Value Area Low and High (simplified version)
    value_area_low = price_levels[int(price_bins * 0.3)]  # Lower 30% of volume profile
    value_area_high = price_levels[int(price_bins * 0.7)]  # Upper 70% of volume profile

    # Add AMT Signal based on Value Area Low and High
    # Assign 1 for buy signal and -1 for sell signal
    df['AMT_signal'] = 0  # Default no signal

    # Buy signal: Price is near the Value Area Low
    df.loc[df['close'] < value_area_low, 'AMT_signal'] = 1

    # Sell signal: Price is near the Value Area High
    df.loc[df['close'] > value_area_high, 'AMT_signal'] = -1

    # Shutdown MT5
    mt5.shutdown()

    return df

def Accuracy(df,ind_count,shift_count) :
    df_1 = df
    # Create columns to track signal outcomes
    df_1['next_close'] = df_1['close'].shift(shift_count)  # Get the next day's close price

    # Buy when Sum_Signal is positive, sell when negative
    df_1['buy_signal'] = (df_1['Sum'] >= ind_count)
    df_1['sell_signal'] = (df_1['Sum'] <= -ind_count)

    # Track if buying or selling leads to a positive/negative price change
    df_1['buy_return'] = (df_1['buy_signal'] & (df_1['next_close'] > df_1['close'])).astype(int)  # 1 if profitable
    df_1['sell_return'] = (df_1['sell_signal'] & (df_1['next_close'] < df_1['close'])).astype(int)  # 1 if profitable
    # Calculate performance metrics
    buy_success_rate = (df_result['buy_return'].sum() / df_result['buy_signal'].sum()) * 100
    sell_success_rate = (df_result['sell_return'].sum() / df_result['sell_signal'].sum()) * 100
    accuracy = ((buy_success_rate + sell_success_rate) / 2)
    
    Sell_True_Count = df_1['sell_signal'].sum()
    Buy_True_Count = df_1['buy_signal'].sum()

    df_sum_accuracy = pd.DataFrame({
        'symbol': [symbol],
        'buy_success_rate': [buy_success_rate],
        'sell_success_rate': [sell_success_rate],
        'Accuracy': [accuracy],
        'Buy_True_Count' : [Buy_True_Count],
        'Sell_True_Count' : [Sell_True_Count],
        
    })
    
    # Return the dataframe with symbol and accuracy columns
    return df_1,df_sum_accuracy

def chart(df_1,symbol, use_candlestick=True):
    df_1 = df_1.set_index('time')  # Ensure time is the index

    buy_times = df_1.index[df_1['buy_signal']]
    sell_times = df_1.index[df_1['sell_signal']]

    if use_candlestick:
        mpf.plot(df_1, type='candle', volume=False, style='charles',
                 title = f"{symbol}",
                 ylabel="Price",
                 figsize=(20, 10),
                 vlines=dict(vlines=buy_times.tolist() + sell_times.tolist(), 
                             colors=['g'] * len(buy_times) + ['r'] * len(sell_times), 
                             linestyle='dashed'), warn_too_much_data=100000)
    else:
        # Simple Line Chart Version
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        plt.figure(figsize=(20, 10))
        plt.plot(df_1.index, df_1['close'], label='Close Price', color='blue', linewidth=2)
        plt.scatter(df_1.index[df_1['buy_signal']], df_1['close'][df_1['buy_signal']], 
                    marker='^', color='green', label='Buy Signal', s=80, edgecolors='black', linewidth=1.5)
        plt.scatter(df_1.index[df_1['sell_signal']], df_1['close'][df_1['sell_signal']], 
                    marker='v', color='red', label='Sell Signal', s=80, edgecolors='black', linewidth=1.5)
        plt.title(f"{symbol}", fontsize=20, fontweight='bold')
        plt.xlabel('Time', fontsize=16)
        plt.ylabel('Price', fontsize=16)
        plt.legend(fontsize=14, loc='best')
        plt.xticks(rotation=45, fontsize=12)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

def place_order(symbol, action, sl_atr, tp_atr, priority=1):
    if not mt5.initialize():
        logging.error("Failed to initialize MT5")
        return

    try:
        if not mt5.symbol_select(symbol, True):
            logging.error(f"Failed to select symbol: {symbol}")
            return

        tick_info = mt5.symbol_info_tick(symbol)
        if not tick_info:
            logging.error(f"Failed to get tick info for {symbol}")
            return

        # Get current price
        price = tick_info.ask if action == "buy" else tick_info.bid
        symbol_info = mt5.symbol_info(symbol)
        point = symbol_info.point
        digits = symbol_info.digits  # Get decimal precision
        min_volume = symbol_info.volume_min

        # Minimum stop level handling (avoid zero values)
        stop_level = max(symbol_info.trade_stops_level * point, 10 * point)

        # Convert ATR-based SL and TP into points
        sl_points = sl_atr / point
        tp_points = tp_atr / point

        # Apply priority multipliers
        if priority == 1:
            sl_points *= 2.5
            tp_points *= 4
        elif priority == 2:
            sl_points *= 1.2
            tp_points *= 1

        # Calculate absolute price levels
        if action == "buy":
            sl_price = price - sl_points * point
            tp_price = price + tp_points * point
        else:
            sl_price = price + sl_points * point
            tp_price = price - tp_points * point

        # Validate SL/TP distances
        if abs(price - sl_price) < stop_level:
            sl_price = price - (stop_level if action == "buy" else -stop_level)
        if abs(price - tp_price) < stop_level:
            tp_price = price + (stop_level if action == "sell" else -stop_level)

        # Round SL & TP to match the symbol's precision
        sl_price = round(sl_price, digits)
        tp_price = round(tp_price, digits)

        # Create order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": max(min_volume, 0.01),
            "type": mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 10,
            "magic": 234000,
            "comment": f"ATR-based {action} (Priority {priority})",
            "type_filling": mt5.ORDER_FILLING_IOC,
            "type_time": mt5.ORDER_TIME_GTC
        }

        result = mt5.order_send(request)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(log_file_path, "a") as log_file:
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                log_file.write(f"[{timestamp}] : Order failed: {result.comment} (Code: {result.retcode})\n")
            else:
                log_file.write(f"[{timestamp}] : Executed {action} {symbol} at {price}, SL={sl_price}, TP={tp_price}\n")

    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
    finally:
        mt5.shutdown()


def generate_signal(row):
    if row['Hammer'] > 0 or row['Bullish_Engulfing']:
        return 1
    elif row['Shooting_Star'] < 0 or row['Bearish_Engulfing']:
        return -1
    elif row['Doji'] != 0:
        return 0  # Doji is uncertain, wait for confirmation
    return 0

def model(symbol , shift_momentum, shift_candle) :
    df_result = fetch_data(symbol, timeframe , n_bars)
    #df_result = df_result.head(100)
    
    df_result['Hammer'] = talib.CDLHAMMER(df_result['open'], df_result['high'], df_result['low'], df_result['close'])
    df_result['Inverted_Hammer'] = talib.CDLINVERTEDHAMMER(df_result['open'], df_result['high'], df_result['low'], df_result['close'])
    df_result['Shooting_Star'] = talib.CDLSHOOTINGSTAR(df_result['open'], df_result['high'], df_result['low'], df_result['close'])
    df_result['Bullish_Engulfing'] = talib.CDLENGULFING(df_result['open'], df_result['high'], df_result['low'], df_result['close']) > 0
    df_result['Bearish_Engulfing'] = talib.CDLENGULFING(df_result['open'], df_result['high'], df_result['low'], df_result['close']) < 0
    df_result['Doji'] = talib.CDLDOJI(df_result['open'], df_result['high'], df_result['low'], df_result['close'])
    df_result['Morning_Star'] = talib.CDLMORNINGSTAR(df_result['open'], df_result['high'], df_result['low'], df_result['close'])
    df_result['Evening_Star'] = talib.CDLEVENINGSTAR(df_result['open'], df_result['high'], df_result['low'], df_result['close'])
    df_result['Piercing_Pattern'] = talib.CDLPIERCING(df_result['open'], df_result['high'], df_result['low'], df_result['close'])
    df_result['Dark_Cloud_Cover'] = talib.CDLDARKCLOUDCOVER(df_result['open'], df_result['high'], df_result['low'], df_result['close'])
    df_result['Three_White_Soldiers'] = talib.CDL3WHITESOLDIERS(df_result['open'], df_result['high'], df_result['low'], df_result['close'])
    df_result['Three_Black_Crows'] = talib.CDL3BLACKCROWS(df_result['open'], df_result['high'], df_result['low'], df_result['close'])
    df_result['Three_Inside_Up'] = talib.CDL3INSIDE(df_result['open'], df_result['high'], df_result['low'], df_result['close']) > 0
    df_result['Three_Inside_Down'] = talib.CDL3INSIDE(df_result['open'], df_result['high'], df_result['low'], df_result['close']) < 0
    #-------------------------------------------------------------------------------------
    df_result['Money_Flow_Multiplier'] = (df_result['close'] - df_result['low']) - (df_result['high'] - df_result['close'])
    df_result['Money_Flow_Multiplier'] /= (df_result['high'] - df_result['low'])
    df_result['Money_Flow_Volume'] = df_result['Money_Flow_Multiplier'] * df_result['volume']
    df_result['CMF'] = df_result['Money_Flow_Volume'].rolling(window=20).sum() / df_result['volume'].rolling(window=20).sum()
    df_result['CMF_signal'] = 0
    df_result.loc[df_result['CMF'] > 0, 'CMF_signal'] = 1  # Buy signal when CMF is positive
    df_result.loc[df_result['CMF'] < 0, 'CMF_signal'] = -1  # Sell signal when CMF is negative
    #-------------------------------------------------------------------------------------
    
    df_result['candle'] = df_result.apply(generate_signal, axis=1)
    df_result['candle'] = df_result['candle'].shift(shift_candle)
    
    df_result['RSI_Signal'] = df_result['RSI_Signal'].shift(shift_momentum)
    df_result['RSI_2_Signal'] = df_result['RSI_2_Signal'].shift(shift_momentum)
    df_result['W%R_Signal'] = df_result['W%R_Signal'].shift(shift_momentum)
    df_result['BB_Signal'] = df_result['BB_Signal'].shift(shift_momentum)
    df_result['Keltner_Signal'] = df_result['Keltner_Signal'].shift(shift_momentum)
    df_result['CCI_Signal'] = df_result['CCI_Signal'].shift(shift_momentum)    # Calculate the Money Flow Volume
    df_result['OBV_Signal'] = df_result['OBV_Signal'].shift(shift_momentum)
    df_result['AMT_signal'] = df_result['AMT_signal'].shift(shift_momentum)
    

    df_result['Sum'] = (
       df_result['candle'] + df_result['Keltner_Signal'] + df_result['CCI_Signal'] #+ df_result['AMT_signal'] 
    )
    
    return df_result

restart_mt5() 
  
timeframe = mt5.TIMEFRAME_H1
n_bars = 250

currency_pairs_all = [
    'EURUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'XAUUSD', 'EURGBP', 'EURJPY', 'EURCHF', 'EURAUD', 
    'EURCAD', 'GBPJPY', 'GBPCHF', 'GBPAUD', 'GBPCAD', 'AUDJPY', 'AUDCHF', 'CHFJPY', 'CADJPY',
    'NZDJPY', 'CADCHF', 'EURNZD', 'USDMXN', 'USDZAR', 'USDSGD', 'USDHKD', 'EURZAR', 'EURSGD', 
    'EURHKD', 'GBPSGD', 'AUDSGD', 'CHFSGD', 'NZDCAD', 'SGDJPY', 'BTCUSD', 'XRPUSD', 'ETHUSD', 
    'US2000', 'US30', 'US500'
]


# "AUDCAD", 'GBPUSD', 'AUDUSD', 'NZDUSD',  'AUDNZD', NZDCHF ,'GBPNZD', 'USDTRY', 
#Long : EURAUD

for symbol in currency_pairs_all:
    try:
        df_result = model(symbol , shift_momentum=4, shift_candle=1)
        latest = df_result.iloc[-1]

        if df_result['Sum'].iloc[-1] == 3 and df_result['Trend_EMA'].iloc[-1] == 1:
            sl_atr = 0.4 * latest['ATR']
            tp_atr = 0.6 * latest['ATR']
            place_order(symbol, 'buy', sl_atr, tp_atr, priority=1)
            print("Buy :", symbol)
            print("Sum :", df_result['Sum'].iloc[-1])

            with open(log_file_path, "a") as log_file:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"[{timestamp}] : Buy : {symbol}  , RSI : {df_result['RSI'].iloc[-1]}\n")

        elif df_result['Sum'].iloc[-1] == -3 and df_result['Trend_EMA'].iloc[-1] == -1:
            sl_atr = 0.4 * latest['ATR']
            tp_atr = 0.6 * latest['ATR']
            place_order(symbol, 'sell', sl_atr, tp_atr, priority=1)
            print("Sell :", symbol)
            print("Sum :", df_result['Sum'].iloc[-1])

            with open(log_file_path, "a") as log_file:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"[{timestamp}] : Sell : {symbol}  , RSI : {df_result['RSI'].iloc[-1]}\n")

        else:
            with open(log_file_path, "a") as log_file:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"[{timestamp}] : Not Order : {symbol}  , Sum: {df_result['Sum'].iloc[-1]}\n")

    except Exception as e:
        with open(log_file_path, "a") as log_file:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"[{timestamp}] : Error processing {symbol} - {str(e)}\n")
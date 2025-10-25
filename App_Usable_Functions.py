def import_data(ticker:str):
    import yfinance as yf
    df=yf.download(ticker,period='max',auto_adjust=False)
    df=df.reset_index()
    df.columns=['Date','Adj Close','Close','High','Low','Open','Volume']
    beginning_cols=['Date','Open','Close','Adj Close']
    later_cols=[c for c in df.columns if c not in beginning_cols]
    df=df[beginning_cols+later_cols]
    return df


import pandas as pd
def feature_engineering(df:pd.DataFrame):
    from ta.trend import MACD
    from ta.momentum import RSIIndicator
    # 1. MACD
    # adds MACD Line,Signal Line and MACD histogram
    macd_indicator=MACD(close=df['Adj Close'],window_fast=12,window_slow=26,window_sign=9,fillna=False)

    df['MACD_Line']=macd_indicator.macd()
    df['MACD_Signal']=macd_indicator.macd_signal()
    df['MACD_Histogram']=macd_indicator.macd_diff()

    # 2. RSI
    rsi_indicator=RSIIndicator(close=df['Adj Close'],window=14,fillna=False)
    df['RSI_14']=rsi_indicator.rsi()

    # 3. Daily % Return
    df['Daily % Return']=100*(df['Close']-df['Close'].shift(1))/df['Close'].shift(1)

    # 4. MA (Moving Averages)
    ma_windows=[10,20,50,100,200]
    for w in ma_windows:
        df[f'SMA_{w}']=df['Adj Close'].rolling(window=w).mean()           # SMA features
        df[f'EMA_{w}']=df['Adj Close'].ewm(span=w,adjust=False).mean()    # EMA features

    df=df.fillna(0)

    # 5. Close Lags
    def create_lags(data,column='Close',lags=[2,5,7,14]):
        for lag in lags:
            data[f'{column}_LAG{lag}']=data[column].shift(lag)
        return data

    lags=[1,2,3,5,7,14]
    df=create_lags(df,column='Close',lags=lags)
    df=df.dropna()
    return df

def model_train(df,n_steps=100,k=2):
    train_size=int(len(df)*0.8)
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler(feature_range=(0,1))
    data=scaler.fit_transform(df[['Adj Close']])
    import numpy as np
    x=[]
    y=[]
    for i in range(n_steps,len(df)-k+1):
        x.append(data[i-n_steps:i])    # past n days
        y.append(data[i:i+k])          # next k days
    x=np.array(x)
    y=np.array(y)
    x=x.reshape((x.shape[0],x.shape[1],1))
    x_train,x_test=x[:train_size],x[train_size-n_steps:]
    y_train,y_test=y[:train_size],y[train_size-n_steps:]

    import tensorflow
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU,Dense
    model=Sequential([
        GRU(80,input_shape=(x_train.shape[1],1),return_sequences=True),
        GRU(60),
        Dense(y_train.shape[1])
    ])
    model.compile(loss='mean_squared_error',optimizer='adam')
    history=model.fit(x_train,y_train,epochs=30,batch_size=32,validation_split=0.2,verbose=-1)
    model.save('models/finsight_gru.h5')
    print("Model trained and saved.")
    return model,scaler,x_test,y_test,k

def predict(model,scaler,x_test,y_test,k):
    y_pred=model.predict(x_test)
    y_final=scaler.inverse_transform(y_pred)
    y_test=scaler.inverse_transform(y_test)
    import matplotlib.pyplot as plt
    plt.plot(y_final,label='Predicted')
    plt.plot(y_test,label='Actual')
    plt.title('Predicted vs Actual')
    plt.legend()
    plt.show()
    from sklearn.metrics import mean_absolute_percentage_error
    mape=mean_absolute_percentage_error(y_test,y_final)*100
    return y_final[-k],mape

model=model_train(feature_engineering(import_data('COCHINSHIP.NS')))

predict(model[0],model[1],model[2],model[3].reshape(model[3].shape[0],model[3].shape[1]),model[4])
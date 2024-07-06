
#**************** IMPORT PACKAGES ********************
from flask import Flask, render_template, request
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime
import math
import yfinance as yf
import nltk
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")
import os
from sklearn.metrics import accuracy_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#***************** FLASK *****************************
app = Flask(__name__)

#To control caching so as to save and retrieve plot figs on client side
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response

@app.route('/data')
def index():
   return render_template('index.html')
def calculate_bollinger_bands(df, window=20):
    # Tính Moving Average (MA) cho window ngày
    df['MA'] = df['Close'].rolling(window=window).mean()

    # Tính độ lệch chuẩn (Standard Deviation - std) cho window ngày
    df['std'] = df['Close'].rolling(window=window).std()

    # Tính dải trên và dải dưới của Bollinger Bands
    df['BB_Upper'] = df['MA'] + (2 * df['std'])
    df['BB_Lower'] = df['MA'] - (2 * df['std'])
    
    # Vẽ biểu đồ
    plt.figure(figsize=(15, 6))  # Đặt kích thước của biểu đồ trước khi vẽ
    plt.plot(df['Close'], label='Close Price')
    plt.plot(df['BB_Upper'], label='Upper Bollinger Band')
    plt.plot(df['BB_Lower'], label='Lower Bollinger Band')

    # Đặt nhãn cho trục x và trục y
    plt.xlabel('Date')
    plt.ylabel('Price')

    # Đặt tiêu đề cho biểu đồ
    plt.title('Stock Price with Bollinger Bands')

    # Thêm chú thích
    plt.legend()

    # Lưu biểu đồ
    plt.savefig('static/Stock_Price_Bollinger_Bands.png')
    plt.close()

    return df
def calculate_b_percent(data):
    b_percent_values = []

    for i in range(len(data)):
        b_percent = (data['Close'][i] - data['BB_Lower'][i]) / (data['BB_Upper'][i] - data['BB_Lower'][i])
        b_percent_values.append(b_percent)

    return b_percent_values
# Hàm tính chỉ số Momentum (MOM)
def calculate_momentum(data, window):
    momentum_values = (data['Close'] / data['Close'].shift(window) - 1) * 100
    plt.figure(figsize=(15, 6)) 
    plt.plot(momentum_values)
    plt.xlabel('Index')
    plt.ylabel('OBV Values')
    plt.title('Momentum (MOM)')
    plt.savefig('static/Momentum.png')
    plt.close()
    return momentum_values
def calculate_daily(data):
    daily = (data['Close'] -data['Close'].shift(1))
    return daily 
def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    # Tính Moving Average Convergence Divergence (MACD)
    short_ema = df['Close'].ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = df['Close'].ewm(span=long_window, min_periods=1, adjust=False).mean()
    df['MACD'] = short_ema - long_ema

    # Tính signal line từ MACD
    df['Signal_Line'] = df['MACD'].ewm(span=signal_window, min_periods=1, adjust=False).mean()
# Vẽ biểu đồ MACD
    plt.figure(figsize=(15, 6)) 
    plt.plot(df.index, df['MACD'], label='MACD', color='blue')
    plt.plot(df.index, df['Signal_Line'], label='Signal Line', color='red')

        # Đặt nhãn cho trục x và trục y
    plt.xlabel('Date')
    plt.ylabel('MACD')

    # Đặt tiêu đề cho biểu đồ
    plt.title('MACD and Signal Line')

    # Thêm chú thích
    plt.legend()

    # Lưu biểu đồ
    plt.savefig('static/MACD.png')
    plt.close()
    return df
#Tương quan giá hôm nay với giá hôm qua
def calculate_changes_prop(df, columns):
    for column in columns:
        changes = []
        for i in range(1, len(df)):
            if df[column][i] > df[column][i-1]:
                changes.append(1)  # Tăng so với hôm trước
            elif df[column][i] < df[column][i-1]:
                changes.append(-1)  # Giảm so với hôm trước
            else:
                changes.append(0)  # Không thay đổi
        df[column + '(T-1)'] = [0] + changes
def calculate_changes_tor(df, column):
    changes = []
    for i in range(1, len(df)-1):
        if df[column][i+1] > df[column][i]:
            changes.append(1)  # Tăng so với hôm trước
        elif df[column][i+1] < df[column][i]:
            changes.append(-1)  # Giảm so với hôm trước
        else:
            changes.append(0)  # Không thay đổi
    df[column + '(T+1)'] = [0] + changes+[0]
def plot_closing_price(df):
    plt.figure(figsize=(15, 6)) 
    plt.plot(df['Close'], label='Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Closing Price')
    plt.legend()
    plt.savefig('static/Closing_Price.png')
    plt.close()
@app.route('/insertintotable',methods = ['POST'])
def insertintotable():
    nm = request.form['nm']
    open_now=request.form['open_now']
    #**************** FUNCTIONS TO FETCH DATA ***************************
    def get_historical(quote):
        end = datetime.now()
        start = datetime(end.year-2,end.month,end.day)
        data = yf.download(quote, start=start, end=end)
        df = pd.DataFrame(data=data)
        df.to_csv(''+quote+'.csv')
        if(df.empty):
            ts = TimeSeries(key='N6A6QT6IBFJOPJ70',output_format='pandas')
            data, meta_data = ts.get_daily_adjusted(symbol='NSE:'+quote, outputsize='full')
            #Format df
            #Last 2 yrs rows => 502, in ascending order => ::-1
            data=data.head(503).iloc[::-1]
            data=data.reset_index()
            #Keep Required cols only
            df=pd.DataFrame()
            df['Date']=data['date']
            df['Open']=data['1. open']
            df['High']=data['2. high']
            df['Low']=data['3. low']
            df['Close']=data['4. close']
            df['Adj Close']=data['5. adjusted close']
            df['Volume']=data['6. volume']
            df.to_csv(''+quote+'.csv',index=False)
        return

    def calculate_obv(df):
        obv_values = [0]  # Giá trị OBV ban đầu là 0
        for i in range(1, len(df)):
            if df['Close'][i] > df['Close'][i-1]:
                obv_values.append(obv_values[-1] + df['Volume'][i])  # Tăng OBV
            elif df['Close'][i] < df['Close'][i-1]:
                obv_values.append(obv_values[-1] - df['Volume'][i])  # Giảm OBV
            else:
                obv_values.append(obv_values[-1])  # OBV không đổi
        obv_values = [x / 10000000 for x in obv_values]  # Chia cho 1000

        # Vẽ biểu đồ OBV
        plt.figure(figsize=(15, 6)) 
        plt.plot(obv_values)
        plt.xlabel('Index')
        plt.ylabel('OBV Values')
        plt.title('On-Balance Volume (OBV)')
        plt.savefig('static/OBV.png')
        plt.close()

        return obv_values
    def logistic(df,open_now):
        
        # Calculate changes for different columns
        calculate_changes_tor(df,'High')
        calculate_changes_tor(df,'Open')
        data=['Open','Close','Volume','MOM','OBV','%B','High','Low','Return_daily']
        calculate_changes_prop(df,data)
        df_subset = df.iloc[20:-1]
        # Prepare features and target variable for modeling
        features = pd.DataFrame({'Open(T-1)': df_subset['Open(T-1)'],'High(T-1)': df_subset['High(T-1)'],'Low(T-1)': df_subset['Low(T-1)'],'Close(T-1)': df_subset['Close(T-1)'], 'OBV(T-1)': df_subset['OBV(T-1)'], '%B(T-1)': df_subset['%B(T-1)'],'Volume(T-1)': df_subset['Volume(T-1)'],'MOM(T-1)': df_subset['MOM(T-1)'],'Open(T+1)': df_subset['Open(T+1)'],'Return_daily(T-1)': df_subset['Return_daily(T-1)']})
        labels = df_subset['High(T+1)']  # We shift -1 to align changes in High with the next day

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        # Initialize and train logistic regression model
        model = LogisticRegression(penalty='l2', random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Evaluate the model
        mse = mean_squared_error(y_test, predictions)
        print("Mean Squared Error:", mse)
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_test, predictions)
        
        # Tạo danh sách các lớp được dự đoán
        predicted_classes = np.unique(predictions)

        # Vẽ ma trận confusion
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=predicted_classes, yticklabels=predicted_classes)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('static/Confusion_matrix.png')  # Lưu hình ảnh ma trận nhầm lẫn

        # Return the accuracy
        # # Nhập dữ liệu từ bàn phím
        data = float(open_now)

        # Kiểm tra điều kiện và gán giá trị cho cột Open_tour của hàng cuối cùng trong DataFrame
        if data > df['Open'].iloc[-1]:
            df.at[df.index[-1], 'Open(T+1)'] = 1
        elif data < df['Open'].iloc[-1]:
            df.at[df.index[-1], 'Open(T+1)'] = -1
        else:
            df.at[df.index[-1], 'Open(T+1)'] = 0

        print( df.at[df.index[-1], 'Open(T+1)'])
        # Lấy hàng cuối cùng của DataFrame df và chọn các cột quan tâm

        latest_data = df.iloc[[-1]][['Open(T-1)', 'High(T-1)','Low(T-1)', 'Close(T-1)', 'OBV(T-1)', '%B(T-1)','Volume(T-1)','MOM(T-1)','Open(T+1)','Return_daily(T-1)']]
        print(latest_data)
        # Sử dụng mô hình đã huấn luyện để dự đoán biến 'Movement' cho dữ liệu cuối cùng
        predictions = model.predict(latest_data)


        return accuracy,predictions


    
        
    #**************GET DATA ***************************************
    quote=nm
    #Try-except to check if valid stock symbol
    try:
        get_historical(quote)
    except:
        return render_template('index.html', not_found=True)
    else:
        #************** PREPROCESSUNG ***********************
        df = pd.read_csv(''+quote+'.csv')
        print("##############################################################################")
        print("Today's",quote,"Stock Data: ")
        today_stock = df.iloc[-1:]
        print(today_stock)
        print("##############################################################################")
        df = df.dropna()
        code_list = []
        for i in range(0, len(df)):
            code_list.append(quote)
        df2 = pd.DataFrame(code_list, columns=['Code'])
        df2 = pd.concat([df2, df], axis=1)
        df = df2
        plot_closing_price(df)
        obv = calculate_obv(df)
        # Tính cột OBV từ dữ liệu DataFrame df
        obv = calculate_obv(df)
        # Thêm cột OBV vào DataFrame
        df['OBV'] = obv
        df['Return_daily']=calculate_daily(df)
        # Thêm cột %B vào DataFrame
        df = calculate_bollinger_bands(df)
        df['%B'] = calculate_b_percent(df)
        # Vẽ biểu đồ
        df['MOM'] = calculate_momentum(df,10)
        df = calculate_macd(df)
        
        acc,predict= logistic(df,open_now)
        # Render template with results
        today_stock = today_stock.round(2)
        return render_template('results.html', quote=quote,acc=round(acc,2)*100,predict=predict,
                               open_s=today_stock['Open'].to_string(index=False),
                               close_s=today_stock['Close'].to_string(index=False),
                               adj_close=today_stock['Adj Close'].to_string(index=False),
                               low_s=today_stock['Low'].to_string(index=False),
                               vol=today_stock['Volume'].to_string(index=False),
                               high_s=today_stock['High'].to_string(index=False),
                               )

if __name__ == '__main__':
    try:
        app.run(debug=False)
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

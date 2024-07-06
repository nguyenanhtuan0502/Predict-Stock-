
#**************** IMPORT PACKAGES ********************
from flask import Flask, render_template, request
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from flask import Flask, render_template, session
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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")
import os
from sklearn.metrics import accuracy_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
#***************** FLASK *****************************
app = Flask(__name__)
app.config['SECRET_KEY'] = '123'
#To control caching so as to save and retrieve plot figs on client side
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
   return render_template('index.html')
import matplotlib.pyplot as plt
import ta

# Hàm tính chỉ số Momentum (MOM)
def calculate_momentum(data, window):
    momentum_values = ta.momentum.roc(data['Close'], window=window)
    return momentum_values
def calculate_percent_b(data, window=20):
    # Sử dụng thư viện TA để tính toán Bollinger Bands
    bollinger_bands = ta.volatility.BollingerBands(close=data["Close"], window=window, window_dev=2)

    # Tính chỉ số %B
    data['%B'] = (data["Close"] - bollinger_bands.bollinger_lband()) / (bollinger_bands.bollinger_hband() - bollinger_bands.bollinger_lband())

    return data
def calculate_percent_b(data, window=20):
    # Tính Moving Average (MA) cho window ngày
    bollinger_bands = ta.volatility.BollingerBands(close=data["Close"], window=window, window_dev=2)

    data['%B'] = (data["Close"] - bollinger_bands.bollinger_lband()) / (bollinger_bands.bollinger_hband() - bollinger_bands.bollinger_lband())
    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))  # Đặt kích thước của biểu đồ trước khi vẽ
    plt.plot(data['Date'], data['Close'], label='Close Price')
    plt.plot(data['Date'], bollinger_bands.bollinger_hband(), label='Upper Bollinger Band')
    plt.plot(data['Date'], bollinger_bands.bollinger_lband(), label='Lower Bollinger Band')

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

    return data

# Hàm tính chỉ số Momentum (MOM)

def calculate_momentum(data, window):
    data['MOM'] = ta.momentum.roc(data['Close'], window=window)
    plt.figure(figsize=(10, 6)) 
    plt.plot(data['Date'], data['MOM'])  # Sử dụng cột 'Date' làm trục hoành
    plt.xlabel('Date')
    plt.ylabel('Momentum Values')
    plt.title('Momentum (MOM)')
    plt.savefig('static/Momentum.png')
    plt.close()
    return data

def calculate_daily(data):
    daily = (data['Close'] -data['Close'].shift(1))
    return daily 
def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    # Tính MACD và đường tín hiệu sử dụng thư viện TA
    df['MACD'] = ta.trend.macd(df['Close'], window_slow=long_window, window_fast=short_window)
    # Vẽ biểu đồ MACD và đường tín hiệu
    plt.figure(figsize=(10, 6)) 
    plt.plot(df['Date'], df['MACD'], label='MACD', color='blue')
    

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

# Sử dụng hàm
#Tương quan giá hôm nay với giá hôm qua
def calculate_changes_prop(df, columns):
    for column in columns:
        changes = []
        y = np.where(df[column] > df[column].shift(1), 1, -1)
        changes = list(y)  # Convert numpy array to a list
        df[column + '(T-1)'] = changes
def calculate_changes_tor_n(df, column, n):
    y = np.where(df[column].shift(-n) > df[column], 1, -1)
    changes = list(y)  # Convert numpy array to a list
    df[column + '(T+' + str(n) + ')'] = changes
def calculate_changes(df, columns):
    for column in columns:
        df[column + '_cor'] = (df[column + '(T-1)'] * df['High(T+1)'])

def xac_xuat(df, columns):
    xacxuat = {}
    if isinstance(columns, str):  # Kiểm tra nếu columns là một chuỗi
        columns = [columns+'_cor']  # Chuyển đổi thành danh sách chứa một phần tử
    for col in columns:
        xacxuat[col] = df[col+'_cor'].sum() / len(df)

    # Vẽ biểu đồ cột
    plt.figure(figsize=(10,6)) 
    plt.bar(xacxuat.keys(), xacxuat.values())
    plt.xlabel('Columns')
    plt.ylabel('Hệ số')
    plt.title('Biểu đồ hệ số tương quan của các chỉ số với giá cao nhất ngày mai')
    plt.xticks(rotation=45)
    # Thêm giá trị lên từng cột
    for col, value in xacxuat.items():
        plt.text(col, value, round(value, 2), ha='center', va='bottom')
    
    # Lưu biểu đồ thành file hình ảnh
    plt.savefig('static/Tuong_Quan.png')



import matplotlib.pyplot as plt

def plot_closing_price(df):
    # Chuyển đổi cột 'Date' sang định dạng datetime nếu cần
    df['Date'] = pd.to_datetime(df['Date'])

    plt.figure(figsize=(10, 6)) 
    plt.plot(df['Date'], df['Close'], label='Close Price')  # Sử dụng cột 'Date' làm trục hoành
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Closing Price')
    plt.legend()
    plt.savefig('static/Closing_Price.png')
    plt.close()
def calculate_williams_percent_r(data, window):
    data['HighestHigh'] = data['High'].rolling(window=window).max()
    data['LowestLow'] = data['Low'].rolling(window=window).min()
    data['%R'] = ((data['HighestHigh'] - data['Close']) / (data['HighestHigh'] - data['LowestLow'])) * (-100)
    data.drop(['HighestHigh', 'LowestLow'], axis=1, inplace=True)  # Xóa cột tạm thời được tạo ra
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['%R'], color='red', label='%R')
    plt.title('Williams %R')
    plt.xlabel('Date')
    plt.ylabel('%R Value')
    plt.legend()
    plt.savefig('static/%R.png')
    plt.grid(True)
    plt.xticks(rotation=45)
    return data
@app.route('/insertintotable',methods = ['POST'])
def insertintotable():
    nm = request.form['nm']
    open_now=request.form['open_now']
   
    #**************** FUNCTIONS TO FETCH DATA ***************************
    data2=['Open','Close','Volume','MOM','Low','%B','High','%R']
    def get_historical(quote):
        start ='2015-01-02'
        end ='2024-04-16'
        # end = datetime.now()
        # start = datetime(end.year-9,end.month,end.day)
        data = yf.Ticker(quote).history(start=start, end=end)
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

    def calculate_obv(data):
        obv = ta.volume.on_balance_volume(data['Close'], data['Volume'])
        data['OBV'] = obv
            
        plt.figure(figsize=(10, 6)) 
        plt.plot(df['Date'], data['OBV'], label='OBV')  # Sử dụng cột 'Date' làm trục hoành
        plt.xlabel('Date')
        plt.ylabel('OBV Values')
        plt.title('On-Balance Volume (OBV)')
        plt.savefig('static/OBV.png')
        plt.close()
        return data
    
    
    
    def desiontree_1(df, open_now):
        # Calculate changes for different columns
        # Prepare features and target variable for modeling
        features = pd.DataFrame({'Close(T-1)': df['Close(T-1)'], '%B(T-1)': df['%B(T-1)'],'%R(T-1)': df['%R(T-1)'],'Open(T+1)': df['Open(T+1)']})
        labels = df['High(T+1)']  # We shift -1 to align changes in High with the next day
        features = features.iloc[:-1]
        labels = labels.iloc[:-1]
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        # Initialize and train Random Forest model
        model = DecisionTreeClassifier(random_state=42)
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
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        # Tạo một hình mới để vẽ Precision, Recall và F1 Score
        plt.figure(figsize=(8, 8))

        # Vẽ biểu đồ đường cho Precision, Recall và F1 Score
        plt.bar(['Precision', 'Recall', 'F1 Score'], [precision, recall, f1], color='blue', alpha=0.7)

        # Thêm giá trị lên mỗi cột
        for i, v in enumerate([precision, recall, f1]):
            plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')

        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.title('Evaluation Metrics')

        # Lưu hình ảnh
        plt.savefig('static/Evaluation_metrics_desion.png')
        predictions = model.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.round(2)
        # Vẽ biểu đồ
        plt.figure(figsize=(15, 6))
        plt.table(cellText=report_df.values,
          colLabels=report_df.columns,
          rowLabels=report_df.index,
          loc='center',
          cellLoc = 'center', 
          colColours=['lightgray']*len(report_df.columns),
          bbox=[0, 0, 1, 1])  # Thiết lập vùng hiển thị của bảng từ (0,0) đến (1,1)
        plt.axis('off')  # Tắt các trục
        plt.title('Classification Report', fontsize=16, fontweight='bold')

# Lưu hình ảnh
        plt.savefig('static/Classification_report_desion.png', bbox_inches='tight', pad_inches=0.1)
   
        # Vẽ ma trận confusion
        plt.figure(figsize=(8, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=predicted_classes, yticklabels=predicted_classes)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('static/Confusion_matrix_Random.png')  # Lưu hình ảnh ma trận nhầm lẫn
        plt.figure(figsize=(13, 8)) 
        plot_tree(model, feature_names=features.columns.tolist(), class_names=[str(c) for c in model.classes_], filled=True, rounded=True, fontsize=10)
        plt.title('Desion Tree')
        plt.savefig('static/Decision_Tree.png')
        predictions = model.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.round(2)
        # Vẽ biểu đồ
        plt.figure(figsize=(15, 6))
        plt.table(cellText=report_df.values,
          colLabels=report_df.columns,
          rowLabels=report_df.index,
          loc='center',
          cellLoc = 'center', 
          colColours=['lightgray']*len(report_df.columns),
          bbox=[0, 0, 1, 1])  # Thiết lập vùng hiển thị của bảng từ (0,0) đến (1,1)
        plt.axis('off')  # Tắt các trục
        plt.title('Classification Report', fontsize=16, fontweight='bold')
     
        # Return the accuracy
        # # Nhập dữ liệu từ bàn phím
        open_now = float(open_now)

        # Kiểm tra điều kiện và gán giá trị cho cột Open_tour của hàng cuối cùng trong DataFrame
        if open_now > df['Open'].iloc[-1]:
            df.at[df.index[-1], 'Open(T+1)'] = 1
        elif open_now< df['Open'].iloc[-1]:
            df.at[df.index[-1], 'Open(T+1)'] = -1
        else:
            df.at[df.index[-1], 'Open(T+1)'] = 0

        print(df.at[df.index[-1], 'Open(T+1)'])
        # Lấy hàng cuối cùng của DataFrame df và chọn các cột quan tâm
        latest_data = df.iloc[[-1]][['Close(T-1)', '%B(T-1)','%R(T-1)','Open(T+1)']]
        print(latest_data)

        # Sử dụng mô hình đã huấn luyện để dự đoán biến 'Movement' cho dữ liệu cuối cùng
        predictions = model.predict(latest_data)

        return accuracy, predictions[0]

    def desiontree_2(df):
        # Prepare features and target variable for modeling
        features = pd.DataFrame({'Close(T-1)': df['Close(T-1)'], '%B(T-1)': df['%B(T-1)'],'%R(T-1)': df['%R(T-1)']})
        labels = df['High(T+1)']  # We shift -1 to align changes in High with the next day
        features = features.iloc[:-1]
        labels = labels.iloc[:-1]
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        # Initialize and train Random Forest model
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        plt.figure(figsize=(13, 8)) 
        plt.title('Desion Tree')
        plot_tree(model, feature_names=features.columns.tolist(), class_names=[str(c) for c in model.classes_], filled=True, rounded=True, fontsize=10)
        plt.title('Desion Tree')
        plt.savefig('static/Decision_Tree.png')
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
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        # Tạo một hình mới để vẽ Precision, Recall và F1 Score
        plt.figure(figsize=(8, 8))

        # Vẽ biểu đồ đường cho Precision, Recall và F1 Score
        plt.bar(['Precision', 'Recall', 'F1 Score'], [precision, recall, f1], color='blue', alpha=0.7)

        # Thêm giá trị lên mỗi cột
        for i, v in enumerate([precision, recall, f1]):
            plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')

        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.title('Evaluation Metrics')

        # Lưu hình ảnh
        plt.savefig('static/Evaluation_metrics_desion.png')
        predictions = model.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.round(2)
        # Vẽ biểu đồ
        plt.figure(figsize=(15, 6))
        plt.table(cellText=report_df.values,
          colLabels=report_df.columns,
          rowLabels=report_df.index,
          loc='center',
          cellLoc = 'center', 
          colColours=['lightgray']*len(report_df.columns),
          bbox=[0, 0, 1, 1])  # Thiết lập vùng hiển thị của bảng từ (0,0) đến (1,1)
        plt.axis('off')  # Tắt các trục
        plt.title('Classification Report', fontsize=16, fontweight='bold')
        plt.savefig('static/Classification_report_desion.png', bbox_inches='tight', pad_inches=0.1)
        # Vẽ ma trận confusion
        plt.figure(figsize=(8, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=predicted_classes, yticklabels=predicted_classes)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('static/Confusion_matrix_Random.png')  # Lưu hình ảnh ma trận nhầm lẫn

        # Return the accuracy
        # # Nhập dữ liệu từ bàn phím
        
        # Lấy hàng cuối cùng của DataFrame df và chọn các cột quan tâm
        latest_data = df.iloc[[-1]][['Close(T-1)', '%B(T-1)','%R(T-1)']]
        print(latest_data)

        # Sử dụng mô hình đã huấn luyện để dự đoán biến 'Movement' cho dữ liệu cuối cùng
        predictions = model.predict(latest_data)

        return accuracy, predictions[0]
        
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
        
        code_list = []
        for i in range(0, len(df)):
            code_list.append(quote)
        df2 = pd.DataFrame(code_list, columns=['Code'])
        df2 = pd.concat([df2, df], axis=1)
        df = df2
        calculate_changes_tor_n(df, 'High',1)
        plot_closing_price(df)    
        # Tính cột OBV từ dữ liệu DataFrame df
        calculate_obv(df)
        df['Return_daily']=calculate_daily(df)
        # Thêm cột %B vào DataFrame
        calculate_percent_b(df, 10)
        calculate_williams_percent_r(df, 14)
        # Vẽ biểu đồ
        calculate_momentum(df,10)
        df = calculate_macd(df)     
        data=['Close','MOM','OBV','%B','%R']
        calculate_changes_prop(df, data)
        calculate_changes_tor_n(df,'Open',1)
        calculate_changes_prop(df,data2)
        calculate_changes(df,data2)
        df = df[(df[['Open','Close(T-1)', '%B(T-1)', 'MOM(T-1)', 'High(T-1)', 'Low(T-1)', 'Volume(T-1)', 'High(T+1)', 'OBV(T-1)']] != 0).all(axis=1)]
        xac_xuat(df,data2)
        
        if open_now != '':
            acc_rand, predict_rand = desiontree_1(df, open_now)
        else:
            acc_rand, predict_rand = desiontree_2(df)
        df_2 = df[['Date','Open','High','Low','Close','Volume','%R','MOM','OBV','%B']]
        print(len(df[df['High(T+1)'] == 1]))
        print(len(df[df['High(T+1)'] == -1]))
        high_cor=df[['Date','Open(T-1)','High(T-1)','Low(T-1)','Close(T-1)','Volume(T-1)','%R(T-1)','MOM(T-1)','OBV(T-1)','%B(T-1)','High(T+1)']]
     
        # Render template with results
        today_stock = today_stock.round(2)
        return render_template('results.html', quote=quote,
                        open_s=today_stock['Open'].to_string(index=False),
                        close_s=today_stock['Close'].to_string(index=False),
                        open_now=open_now,
                        low_s=today_stock['Low'].to_string(index=False),
                        vol=today_stock['Volume'].to_string(index=False),
                        high_s=today_stock['High'].to_string(index=False),
                        acc_rand=round(acc_rand, 3) , predict_rand=predict_rand, df=df_2,high_cor=high_cor)


if __name__ == '__main__':
    try:
        app.run(debug=False)
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

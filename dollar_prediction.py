import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    trainX = np.array(dataX)
    trainY = np.array(dataY)
    return trainX, trainY

def create_baseline(dataset, look_back=1):
    baseline = []
    for i in range(len(dataset)-look_back):
        baseline.append(dataset[i+look_back-1])
    baseline = np.array(baseline)
    return baseline

def split_dataset(dataset, percent=0.8):
    train_size = int(len(dataset) * percent)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    return train, test

def reshape_for_lstm(trainX, testX, look_back):
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    return trainX, testX

def normalize(dataset, a=0, b=1):
    scaler = MinMaxScaler(feature_range=(a, b))
    dataset = scaler.fit_transform(dataset)
    return scaler, dataset

def invert(trainPredict, trainY, testPredict, testY, scaler):
    # invert predictions
    trainPredictInv = scaler.inverse_transform(trainPredict)
    trainYInv = scaler.inverse_transform(trainY)
    testPredictInv = scaler.inverse_transform(testPredict)
    testYInv = scaler.inverse_transform(testY)
    return trainPredictInv, trainYInv, testPredictInv, testYInv

def score(trainPredict, trainY, testPredict, testY, trainBaseline, testBaseline, err_type):
    compute = None
    if err_type == 'mape':
        compute = mean_absolute_percentage_error
    elif err_type == 'mae':
        compute = mean_absolute_error
    elif err_type == 'rmse':
        compute = root_mean_squared_error
    else:
        return
    
    trainScore = compute(trainPredict, trainY)
    testScore = compute(testPredict, testY)
    trainBaselineScore = compute(trainBaseline, trainY)
    testBaselineScore = compute(testBaseline, testY)
    
    print('Train Score: %.5f %s' % (trainScore, err_type.upper()))
    print('Test Score: %.5f %s' % (testScore, err_type.upper()))
    print('Train Baseline Score: %.5f %s' % (trainBaselineScore, err_type.upper()))
    print('Test Baseline Score: %.5f %s' % (testBaselineScore, err_type.upper()))

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def root_mean_squared_error(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def split_baseline(baseline, trainPredict, look_back):
    trainBaseline = baseline[0:trainPredict.shape[0],:]
    testBaseline = baseline[trainPredict.shape[0]+look_back:len(baseline),:]
    return trainBaseline, testBaseline

def createPredictPlot(trainPredict, testPredict, look_back):
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+look_back+1:len(dataset)-look_back+1, :] = testPredict
    return trainPredictPlot, testPredictPlot

def show_plot(trainPredictPlot, testPredictPlot, dataset, look_back):
    # plot baseline and predictions
    plt.figure(figsize=(15, 10))
    plt.plot(dataset)
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()

def show_fig(dataset, trainPredictPlot, testPredictPlot, baseline, look_back):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=dataset[:,0], mode='lines', name='original'))
    fig.add_trace(go.Scatter(y=trainPredictPlot[:,0], mode='lines', name='train'))
    fig.add_trace(go.Scatter(y=testPredictPlot[:,0], mode='lines', name='test'))
    fig.add_trace(go.Scatter(y=baseline[look_back:,0], mode='lines', name='baseline'))
    fig.update_layout(showlegend=True)
    fig.show()

# fix random seed for reproducibility
np.random.seed(7)

# load the dataset
df = pd.read_csv('exchange_rate.csv')
df = df.rename(columns={'Unnamed: 0': 'Date'})
df = df.loc[:,['Date', 'PLN']]

dataset = df['PLN'].values
dataset = dataset.reshape(dataset.shape[0],1)

train, test = split_dataset(dataset, percent=0.8)

scaler, train = normalize(train, 0, 1)
test = scaler.transform(test)

# reshape into X=t and Y=t+1
look_back = 10
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX, testX = reshape_for_lstm(trainX, testX, look_back)

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(15, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer='adam')
model.fit(trainX, trainY, epochs=1000, batch_size=32, verbose=2)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict, trainY, testPredict, testY = invert(trainPredict, trainY, testPredict, testY, scaler)

baseline = create_baseline(dataset, look_back)

trainBaseline, testBaseline = split_baseline(baseline, trainPredict, look_back)

score(trainPredict, trainY, testPredict, testY, trainBaseline, testBaseline, 'mape')

trainPredictPlot, testPredictPlot = createPredictPlot(trainPredict, testPredict, look_back)
show_plot(trainPredictPlot, testPredictPlot, dataset, look_back)
show_fig(dataset, trainPredictPlot, testPredictPlot, baseline, look_back)





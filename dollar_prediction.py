import numpy as np
import pandas as pd
import math
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    trainX = np.array(dataX)
    trainY = np.array(dataY)
    return trainX, trainY.reshape(trainY.shape[0], 1)

def create_baseline(dataset, look_back=1):
    baseline = []
    for i in range(len(dataset)-look_back):
        baseline.append(dataset[i+look_back-1, 0])
    baseline = np.array(baseline)
    return baseline.reshape(baseline.shape[0], 1)

def split_dataset(dataset, baseline, look_back, percent=0.8):
    train_size = int(len(dataset) * percent)
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    trainBaseline = baseline[0:train_size-look_back,:]
    testBaseline = baseline[train_size:len(baseline),:]
    return train, test, trainBaseline, testBaseline

def invert(trainPredict, trainY, testPredict, testY, scaler):
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

def mean_absolute_percentage_error(y_pred, y_true):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.

def root_mean_squared_error(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def createPredictPlot(trainPredict, testPredict, baseline, look_back):
    trainPredictPlot = np.empty_like(dataset[:,0])
    trainPredictPlot = trainPredictPlot.reshape(trainPredictPlot.shape[0],1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict


    testPredictPlot = np.empty_like(dataset[:,0])
    testPredictPlot = testPredictPlot.reshape(testPredictPlot.shape[0],1)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+2*look_back:len(dataset), :] = testPredict
    
    baselinePredictPlot = np.empty_like(dataset[:,0])
    baselinePredictPlot = baselinePredictPlot.reshape(baselinePredictPlot.shape[0],1)
    baselinePredictPlot[:, :] = np.nan
    baselinePredictPlot[look_back:len(baseline)+look_back, :] = baseline
    return trainPredictPlot, testPredictPlot, baselinePredictPlot

def show_fig(dataset, trainPredictPlot, testPredictPlot, baselinePredictPlot, look_back):
    fig = go.Figure()
    print(look_back)
    fig.add_trace(go.Scatter(y=dataset[:,0], mode='lines', name="przebieg oryginalny"))
    fig.add_trace(go.Scatter(y=trainPredictPlot[:,0], mode='lines',\
                             name='przebieg przewidziany przez model na zbiorze treningowym'))
    fig.add_trace(go.Scatter(y=testPredictPlot[:,0], mode='lines',\
                             name='przebieg przewidziany przez model na zbiorze testowym'))
    fig.add_trace(go.Scatter(y=baselinePredictPlot[:,0], mode='lines', name='przebieg baseline\'u'))
    fig.update_layout(showlegend=True, xaxis_title="dzień",\
                      yaxis_title="kurs dolara", title="Wykres kursu dolara")
    fig.show()

np.random.seed(7)

df = pd.read_csv('exchange_rate.csv')
df = df.iloc[::-1].reset_index()

df = df.rename(columns={'Unnamed: 0': 'Date'})
df = df.loc[:,['Date', 'PLN']]
s = pd.to_datetime(df['Date'])
df['day of week'] = s.dt.dayofweek
s = pd.get_dummies(df['day of week'])
df = pd.concat([df, s], axis=1)
df = df.drop(['day of week'], axis=1)
df = df.drop(['Date'], axis=1)

dataset = df.values
look_back = 10
baseline = create_baseline(dataset, look_back)
train, test, trainBaseline, testBaseline = split_dataset(dataset, baseline, look_back, percent=0.8)
shape_train = (len(train[:, 0]), 1)
shape_test = (len(test[:, 0]), 1)
scaler = MinMaxScaler(feature_range=(0, 1))
train[:, 0] = scaler.fit_transform(np.reshape(train[:, 0], shape_train)).squeeze()
test[:, 0] = scaler.transform(np.reshape(test[:, 0], shape_test)).squeeze()

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

model = Sequential()
optimizer = keras.optimizers.Adam(lr=0.001)
model.add(LSTM(15, input_shape=(look_back, 8)))
model.add(Dense(1))
model.compile(loss='mae', optimizer=optimizer)
model.fit(trainX, trainY, epochs=1000, batch_size=32, verbose=2)

K.set_value(model.optimizer.learning_rate, 0.0001)
model.fit(trainX, trainY, epochs=500, batch_size=32, verbose=2)

K.set_value(model.optimizer.learning_rate, 0.00005)
model.fit(trainX, trainY, epochs=500, batch_size=32, verbose=2)

model.save_weights('model.h5')

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict, trainY, testPredict, testY = invert(trainPredict, trainY, testPredict, testY, scaler)

score(trainPredict, trainY, testPredict, testY, trainBaseline, testBaseline, 'mape')
print('\n')
score(trainPredict, trainY, testPredict, testY, trainBaseline, testBaseline, 'mae')
print('\n')
score(trainPredict, trainY, testPredict, testY, trainBaseline, testBaseline, 'rmse')

trainPredictPlot, testPredictPlot, baselinePredictPlot =\
    createPredictPlot(trainPredict, testPredict, baseline, look_back)
show_fig(scaler.inverse_transform(dataset), trainPredictPlot, testPredictPlot,\
         baselinePredictPlot, look_back)

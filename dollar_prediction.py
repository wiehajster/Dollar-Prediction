#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go


# In[3]:


# fix random seed for reproducibility
np.random.seed(7)


# In[5]:


# load the dataset
df = pd.read_csv('exchange_rate.csv')


# In[9]:


df = df.rename(columns={'Unnamed: 0': 'Date'})
df = df.loc[:,['Date', 'PLN']]
df


# In[16]:


dataset = df['PLN'].values
dataset = dataset.reshape(dataset.shape[0],1)
print(dataset)
print('Shape: ', dataset.shape)
print('Type: ', dataset.dtype)


# In[17]:


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
dataset


# In[18]:


# split into train and test sets
percent = 0.67
train_size = int(len(dataset) * percent)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# In[28]:


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY, baseLine = [], [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    trainX = np.array(dataX)
    trainY = np.array(dataY)
    trainY = trainY.reshape(trainY.shape[0], 1)
    return trainX, trainY


# In[29]:


# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# In[102]:


def create_baseline(dataset, look_back=1):
    baseline = []
    invDataset = scaler.inverse_transform(dataset)
    for i in range(len(dataset)-look_back):
        baseline.append(invDataset[i+look_back, 0])
    baseline = np.array(baseline)
    baseline = baseline.reshape(baseline.shape[0], 1)
    return baseline


# In[30]:


# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# In[31]:


# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# In[34]:


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)


# In[36]:


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredictInv = scaler.inverse_transform(trainPredict)
trainYInv = scaler.inverse_transform(trainY)
testPredictInv = scaler.inverse_transform(testPredict)
testYInv = scaler.inverse_transform(testY)
# calculate root mean squared error
#trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
#print('Train Score: %.2f RMSE' % (trainScore))
#testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
#print('Test Score: %.2f RMSE' % (testScore))


# In[46]:


def mean_absolute_percentage_error(y_true, y_pred):
    #y_true[y_true == 0.0] = 1e-20
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[124]:


trainScore = mean_absolute_percentage_error(trainPredictInv, trainYInv)
print('Train Score: %.2f MAPE' % (trainScore))
testScore = mean_absolute_percentage_error(testPredictInv, testYInv)
print('Test Score: %.2f MAPE' % (testScore))
#baseLine
trainBaseline = baseline[0:trainPredictInv.shape[0],:]
testBaseline = baseline[trainPredictInv.shape[0]:len(baseline),:]
print(testBaseline.shape)

trainScore = mean_absolute_percentage_error(trainBaseline, trainYInv)
print('Train BaseLine Score: %.2f MAPE' % (trainScore))
testScore = mean_absolute_percentage_error(testBaseline, testYInv)
print('Test BaseLine Score: %.2f MAPE' % (testScore))

trainScore = math.sqrt(mean_squared_error(trainPredictInv, trainYInv))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testPredictInv, testYInv))
print('Test Score: %.2f RMSE' % (testScore))

#baseLine
trainScore = math.sqrt(mean_squared_error(trainBaseline, trainYInv))
print('Train Baseline Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testBaseline, testYInv))
print('Test Baseline Score: %.2f RMSE' % (testScore))


# In[82]:


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredictInv)+look_back, :] = trainPredictInv
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredictInv)+look_back+1:len(dataset), :] = testPredictInv
# plot baseline and predictions
plt.figure(figsize=(15, 10))
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[107]:


fig = go.Figure()
fig.add_trace(go.Scatter(y=df.PLN, mode='lines', name='original'))
fig.add_trace(go.Scatter(y=trainPredictPlot[:,0], mode='lines', name='train'))
fig.add_trace(go.Scatter(y=testPredictPlot[:,0], mode='lines', name='test'))
fig.add_trace(go.Scatter(y=baseline[:,0], mode='lines', name='baseline'))
fig.update_layout(showlegend=True)
fig.show()


# In[103]:


baseline = create_baseline(dataset)


# In[104]:


baseline.shape


# In[105]:


baseline


# In[106]:


invDataset = scaler.inverse_transform(dataset)
invDataset


# In[120]:


trainPredictInv.shape[0] + testPredictInv.shape[0]


# In[ ]:





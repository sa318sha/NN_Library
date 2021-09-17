
from Library.Model import Sequential
from Library.Metrics import metrics
from Library.Optimizer import Adam, Mini_Batch_Gradient_Descent
from Library.Layers import Layer_Dense
import numpy as np
import pandas as pd
np.random.seed(0)

# to-do:
# implement regularization
# implement dropout 
# complete the other activation functions
# exponential weighted averages
# learn more about optimizers


# df = pd.read_csv('Data/breastCancer.csv')
# print(df.head())
# # print(df[:,1:4])
# # print(df.isnull().count())
# y = df['diagnosis']
# X = df.iloc[:,2:-1]

# # print(X)
# X = ((X- X.mean())/X.std()).to_numpy()
# y = y.to_numpy()
# # print(X)
# # print('y',y)
# # print('y new', pd.factorize(y)[0])
# y = pd.factorize(y)[0]
# # print(type(y))

X = np.random.randn(3,4)

target = np.array([[0,1,0],[0,0,1],[0,0,1]])

Model = Sequential([
  Layer_Dense(4,10),
  Layer_Dense(10,3, 'softmax')
])

Model.compile(loss_function= metrics.categorical_crossEntropy, metrics=[metrics.accuracy], optimizer= Adam(0.01))

Model.fit(10,20,X,target,validation_split=0)

# breastCancerPredictionModel.evaluation()
# splitting 20 percent for testing

# model1 = Sequential([
#   Layer_Dense(4,3),
#   Layer_Dense(3,3),
#   Layer_Dense(3,3,'softmax'),
#   ])


# model1.forward(X)
# model1.getFinalOutput()
# print('final output',model1.output)
# model1.compile(0.01, loss_function = metrics.categorical_crossEntropy, metrics = [metrics.accuracy])
# model1.fit(50,10,X,target)

# model1.display_results()

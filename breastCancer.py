
from Library.Models.Sequential import Sequential
from Library.Metrics import metrics
from Library.Optimizers.Adam import Adam
from Library.Layers.Layer_Dense import Layer_Dense
import numpy as np
import pandas as pd
np.random.seed(0)

# to-do:
# implement regularization
# implement dropout 
# complete the other activation functions
# learn metafiles and h5 files 
# learn saving and loading data
# learn interaction with databases with python
# learn how to connect js with python or how to build nn on python
# learn open gym 
# learn how to use decorators and vlass methods to improve performance
# mae metrics
# nicholas renotte has a vid on learning agents with just keras and tensorflow
# learn how to use classmethods/property and decorators this is huge topic so once im done with neural networks or im happy with my progress

# exponential weighted averages - finished
# learn more about optimizers - finished


df = pd.read_csv('Data/breastCancer.csv')
print(df.head())
# print(df[:,1:4])
# print(df.isnull().count())
y = df['diagnosis']
X = df.iloc[:,2:-1]

# print(X)
X = ((X- X.mean())/X.std()).to_numpy()
y = y.to_numpy()
# print(X)
# print('y',y)
# print('y new', pd.factorize(y)[0])
y = pd.factorize(y)[0]
# print(type(y))
breastCancerPredictionModel = Sequential([
  Layer_Dense(30,20),
  Layer_Dense(20,10),
  Layer_Dense(10,2, 'softmax')
])

breastCancerPredictionModel.compile(loss_function= metrics.categorical_crossEntropy, metrics=[metrics.accuracy], optimizer= Adam(0.01))

# breastCancerPredictionModel.fit(30,20,X,y,validation_split=0.25)


userChoice = -1
# print('weights',breastCancerPredictionModel.Layers[0].weights)
while (userChoice != 0 ):
  userChoice = input('\n1. for one iteration\n2. for x amount of iteration\n3. Save weights and biasis\n4. Load Weights and Biasis\n0. To Quit\n')
  if(userChoice == '1'):
  # print('hello')
    print('\nAdditional One iteration:\n')
    breastCancerPredictionModel.fit(1,20,X,y,validation_split=0.25)

  if(userChoice == '2'):
    numberOfIterations = int(input('\nnumber of iterations'))
    breastCancerPredictionModel.fit(numberOfIterations,20,X,y,validation_split=0.25)

  if(userChoice == '3'):
    pass
    
  if(userChoice == '4'):
    pass
  if(userChoice == '5'):
    pass

  if(userChoice == '0'):
    break           
# userChoice = input('Enter your choice:\n   1.for Layer 1 weights\n   2.for Layer 2 weights\n   3.for Layer 1 biasis\n   4.for Layer 2 biasis\n   5. for next iteration\n   0. To Quit\n')
# # print('userCHoice', userChoice, type(userChoice))

# if(userChoice == '1'):
#   # print('hello')
#   print('\nlayer 1 weights:\n')
#   print(layer1.weights)

# if(userChoice == '2'):
#   print('\nlayer 2 weights:\n')
#   print(layer2.weights)

# if(userChoice == '3'):
#   print('\nlayer 1 biasis:\n')
#   print(layer1.biases)
  
# if(userChoice == '4'):
#   print('\nlayer 2 biasis:\n')
#   print(layer2.biases)
# if(userChoice == '5'):
#   pass

# if(userChoice == '0'):
#   break
  

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

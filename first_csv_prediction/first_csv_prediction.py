"""
def mae(y_true, y_pred):  # mean absolute error
    return (y_true - y_pred).abs().mean()
"""

# technique for doing an experiment
# we're not actually training anything, just using the line function above

# steps:
# load the data
# make a variable for all x values and one for all y values (get the right column)
# use a for loop and the kfold function from sklearn
# each time in the loop. you'll have:
#   -X_train, X_test, Y_train, Y_test
#   -to be sure, Y_test are the true values, and x_test is all the true x's
#   -so run the line function on the x test inputs
#   -compare with mae function the predictions with Y_test
# average mae's across all folds

# class model
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

df = pd.read_csv('male_heights.csv')
x = df['Age (in months)']
y_true = df['50th Percentile Length (in centimeters)']

y_true = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
# print(x) #need to reshape the input so that each value is isolated in brackets (unsqueeze)

kf = KFold(n_splits=5, random_state=None)
# cnt = 1
cross_val = 0

for train_index, test_index in kf.split(x, y_true):
    x_train, x_test = x[train_index], x[test_index]
    # print(x_test)
    y_train, y_test = y_true[train_index], y_true[test_index]
    # print(y_train)

    # reset the model
    model = nn.Linear(1, 1)  # first is # inputs, second is # outputs #print(model) print(list(model.parameters()))
    loss = nn.MSELoss()  # print(loss)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # print(optimizer)

    for epoch in range(1000):  # train the model
        y_pred = model(x_train)  # run the model on all the training inputs
        epoch_loss = loss(y_pred, y_train)  # compare training predictions to true values
        # print("Loss:", epoch_loss)  # the error
        # magic step
        optimizer.zero_grad()  # clear out the gradient, start from scratch
        # these two below are the machine learning steps
        epoch_loss.backward()  # propagate error backwards (for each weight)--you want to know how much each weight
        # contributed to the error (puts data on each weight)
        optimizer.step()  # tell the optimizer to use that information to go again--meaning figure out which
        # direction to go in and move all the weights accordingly(slightly adjusts all the weights)

        # once model at its best, test your model on the test data and get the loss for the model
        # this part needs to be in the loop, so it remembers what the optimizer has done?
        #if epoch == 999:
    y_pred = model(x_test)
    epoch_loss = loss(y_pred, y_test)
    print("Loss:", epoch_loss)
    cross_val += epoch_loss  # smallest loss

print("Cross Validation Score: ", cross_val / 5)
# cross-validation
# print(list(model.parameters())) #the line function

# task: put in the cross validation, separate train and test data (retrain the model for each portion of the data)
# 4/5 data is training, 1/5 is test
# train on training, validate on test

# torch.nn (neural network library)
# model = Linear(1,1)
# start with a random w&b
# make a loss function (aka error)
# can't take derivative of absolute value, so we don't want MAE
# user MSE instead (mean squared error)-- (sum from i to n (true-pred)^2)/n
# --if you want the units to be right, take square root of the whole thing
# and that is the RMSE
# can get the slope from the MSE
# make an optimizer
# best parameters are where the minimum loss is--visualize graph--
# x's are options for the weights
# y is loss--calculate the slope of the curve
# --want to go where the lower slope is (call this gradient descent)
# --there are local and global minimums--you want to global one
# need to make sure it doesnt stop on the local
# how to do this? randomly restart at a new point and see which direction to go in
# the random restarts make it a stochastic gradient descent --this is the optimizer
# called SGD

# optimizer = SGD(model, lr = 0.001 #should be very small) #lr is the learning rate--how big of a jump to make down
# the gradient

# python is slow
# coding in python but its actually just a wrapper
# there is other languages under this--for ML its C/C++ and nvidia (this is faster)

# on neural networks, you can add layers to be able to solve more complex problems (but won't necessarily predict any better)
    #becuase if you add one neuron after another in a layer it basically reduces to a single weight and bias--not really helping
        #this is in the linear case, but if you use the sigmoid function, it doesnt collapse into a single function like this
        # so a nonlinear function is essential to make this capable of learning anything more complicated than a single simple line
# more parameters to learn
# hyperparameters are things that you have to decide--how many neurons, how many layers, the learning rate, etc
# hyperparameter search: systematically trying things--using every combo of parameters
    #Automated ML, wandb.org (tells you what hyperparameters to try next), etc are tools


""" FIRST RUN PRACTICE CROSS-VAL
import pandas as pd
df = pd.read_csv('/Users/skyler/Desktop/AI/male_heights.csv')
x = df['Age (in months)'].to_list()
y = df['50th Percentile Length (in centimeters)'].to_list()

import numpy as np
ave_pred = 0

for test_set in range(5):
    y_split = np.array_split(y, 5)
    x_split = np.array_split(x, 5)

    y_test = y_split[test_set] #1/5 data
    y_split.pop(test_set)  #4/5 data leftover for y_train
    #y_train = [element for innerList in y_split for element in innerList]
    #y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)

    x_test = x_split[test_set]  # 1/5 data
    x_split.pop(test_set)  # 4/5 data leftover for y_train
    #x_train = [element for innerList in x_split for element in innerList]
    #x_train = torch.tensor(x_train)
    x_test = torch.tensor(x_test)

    #train the model
        #(this is just our equation, not actually training it this time)
    #validate with the test set

    y_pred = line(x_test) #predict y_tests for each x_tests
    print("Y_Prediction:",y_pred)
    ave_pred += int(mae(y_test, y_pred)) #compare predictions to actual
    print(int(mae(y_test, y_pred)))

print("Average:",ave_pred/5)





import torch
import math

from sklearn import model_selection


def line(x):  # will be a tensor (every element of x will get the math done on it, arrays dont work like that)
    # return 10 * np.log(x+3) + 40
    return 1.5 * x + 40

 x = torch.arange(1, 10, 0.1)
 making a tensor #arange function --array range prints 1-10
 3rd value is how much to increase by
 an array, a list, a tensor
 tensor--can be arrays, vectors, scalar(a single number), can contain other tensors(2D, 3D, 4D+ array)
 tensors used in AI instead of arrays

 print(line(x))
 print(x)
 print(x.shape)  # number of elements
 x2 = torch.reshape(x, (10, 9))  # 10 tensors with 9 elements each
 print(line(x2))
 print(x2)
 print(x2.shape)  # torch size is [10, 9]

 import matplotlib.pyplot as plt

 plt.plot(x, line(x))
 plt.show()

 import pandas as pd

 df = pd.read_csv('/Users/skyler/Desktop/AI/male_heights.csv')
 print(df)
 print(df.columns)
 x = df['Age (in months)']
 y_true = df['50th Percentile Length (in centimeters)']
 print(x)
 print(y_true)
"""

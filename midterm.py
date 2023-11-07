import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

df = pd.read_csv('/Users/skyler/Desktop/garments_worker_productivity.csv')
y = df['actual_productivity']
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
print(y.shape)
print(y.dtype)

department = pd.get_dummies(df['department'], dtype=float)
x1 = torch.tensor(department.values, dtype=torch.float32)
print(x1.shape)
print(x1.dtype)

incentive = df['incentive'] / 100
x2 = torch.tensor(incentive, dtype=torch.float32).unsqueeze(1)
print(x2.shape)
print(x2.dtype)

target = df['targeted_productivity']
x3 = torch.tensor(target, dtype=torch.float32).unsqueeze(1)
print(x3.shape)
print(x3.dtype)

team = pd.get_dummies(df['team'], dtype=float)
x4 = torch.tensor(team.values, dtype=torch.float32)
print(x4.shape)
print(x4.dtype)

# num_workers = df['no_of_workers']

x = torch.cat((x1, x2, x3, x4), dim=1)
kf = KFold(n_splits=5, shuffle = True, random_state=42)
cross_val = 0
input_size = x.shape[1]
print(input_size)
print(x.dtype)

def create_model(input_dim):
    model = nn.Sequential(
        nn.Linear(input_dim, 100),
        nn.ReLU(), # not adding parameters, no inputs when adding activation function between linear layers
        nn.Linear(100, 1),
    )
    model = model.to(torch.float32)
    return model

"""
# does it make sense to combine so many variables? are they comparable?
for train_index, test_index in kf.split(x, y):
    x_train, x_test = x[train_index], x[test_index]  # think about this
    print(x_train.dtype)
    y_train, y_test = y[train_index], y[test_index]

    #model = nn.Linear(input_size, 1)  # first is # inputs, second is # outputs #print(model) print(list(model.parameters()))
    model = create_model(input_size)
    loss = nn.MSELoss()  # print(loss)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # print(optimizer)
    train_loss_list = []
    test_loss_list = []

    loss_ave = 0
    train_loss_ave = 0

    for epoch in range(1000):  # train the model
        y_pred = model(x_train)
        epoch_loss = loss(y_pred, y_train)
        train_loss_list.append(epoch_loss.item())
        train_loss_ave += epoch_loss  # contains 1000 summed values
        train_loss_ave /= 1000  # divide by 1000 for average
        optimizer.zero_grad()
        epoch_loss.backward()
        optimizer.step()

        # after the update, test with the new slightly adjusted parameters to track progress
        y_pred = model(x_test)
        # print(y_pred)
        epoch_loss = loss(y_pred, y_test)
        test_loss_list.append(epoch_loss.item())
        # print("Loss:", epoch_loss)
        loss_ave += epoch_loss # contains 1000 summed values
        loss_ave /= 1000 # divide by 1000 for average

    print("Train Average Loss:", train_loss_ave)
    print("Average Loss:", loss_ave)
    cross_val += loss_ave # add average test loss for this fold
    import matplotlib.pyplot as plt
    plt.plot(train_loss_list,  label = 'Train Loss')
    plt.plot(test_loss_list, label= 'Test Loss')
    plt.legend()
    plt.show()
    # add in Adam instead of SGD

print("Cross Validation Score: ", cross_val / 5)
"""

# separate 80/20 train ans test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)
model = create_model(input_size)
loss = nn.MSELoss()  # print(loss)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # print(optimizer)
# not stochastic gradient descent, and don't need as many epochs
train_loss_list = []
test_loss_list = []

for epoch in range(100):  # train the model
    y_pred = model(x_train)
    epoch_loss = loss(y_pred, y_train)
    train_loss_list.append(epoch_loss.item())
    optimizer.zero_grad()
    epoch_loss.backward()
    optimizer.step()

    # after the update, test with the new slightly adjusted parameters to track progress
    y_pred = model(x_test)
    print(y_pred)
    epoch_loss = loss(y_pred, y_test)
    test_loss_list.append(epoch_loss.item())
    print("Loss:", epoch_loss)

import matplotlib.pyplot as plt
plt.plot(train_loss_list,  label = 'Train Loss')
plt.plot(test_loss_list, label= 'Test Loss')
plt.legend()
plt.show()

# make predictions
pred = model(x_test) # multiply by whatever to get form you want
y_test = y_test # multiply by whatever
print(pred.shape)
print(y_test.shape)

# plot predictions vs actual
plt.scatter(y_test.detach(), pred.detach())
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

# plot predictions vs actual as line
plt.plot(y_test.detach(), label = 'Actual')
plt.plot(pred.detach(), label = 'Predicted')
plt.legend()
plt.show()

# show predictions vs actual as a dataframe
df_pred = pd.DataFrame(torch.cat([y_test.detach(), pred.detach()], dim = 1).numpy(), columns = ['actual', 'predicted'])
df_pred['diff'] = df_pred['predicted']-df_pred['actual']

# compute mean squared error
print(df_pred['diff'].pow(2).mean())
# root mean squared error
print(df_pred['diff'].pow(2).mean() **0.5)
# root mean as percentage of mean
print(df_pred['diff'].pow(2).mean() **0.5/df_pred['actual'].mean()) # percent above below actual
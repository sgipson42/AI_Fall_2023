import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import KFold

df = pd.read_csv("male_female_names.csv")

# to predict
gender = df['Gender']
gender_onehot = pd.get_dummies(gender, dtype = float)
gender_encoded = gender_onehot.values
print(gender_encoded)
y = torch.tensor(gender_encoded, dtype = torch.float32)


names = df['Name']
list_names = []
for name in names:
    name = list(name)
    name.reverse()
    list_names.append(name)

# one hot encoding
#print(list_names)
#names_onehot = pd.get_dummies(pd.Series(list_names).explode()).groupby(level=0).sum()
#print(names_onehot)

#
rows = []
for name in list_names:
    vec = []
    for c in name[:10]:
        i = ord(c) # position of onehot for character
        v = [0]*(i-1)+[1]+[0]*(256-i) # vector 128 large, one hot character
        vec.append(v)
    for i in range(10-len(name)):
        v = [0]*256
        vec.append(v)
    rows.append(vec)

t = torch.tensor(rows)
print(t)
print(t.shape)

kf = KFold(n_splits=5, shuffle = True, random_state=42)

for train_index, test_index in kf.split(x, y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(x_train)
    print(y_train)
    model = nn.Sequential(
        nn.Linear(x.shape[1], 100),
        nn.ReLu(),
        nn.Linear(100, 2)
    )
    loss = nn.MAEloss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        y_pred = model(x_train)
        epoch_loss = loss(y_pred, y_train)
        # train_loss_list.append(epoch_loss.item())
        optimizer.zero_grad()
        epoch_loss.backward()
        optimizer.step()

        y_pred = model(x_test)
        print(y_pred)
        epoch_loss = loss(y_pred, y_test)
        #test_loss_list.append(epoch_loss.item())
        print("Loss:", epoch_loss)

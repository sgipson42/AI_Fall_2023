# currently getting nans as predictions-- find them or learning rate?
# this is task specific architecture
# fine-tuned models eliminate the need for task specfic architechture
    # read paper llms are few shot learners
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

df = pd.read_csv("/Users/skyler/Desktop/AI/video_games_sales.csv")
# don't do anything to year of release? or divide by max?
x = torch.tensor(df['Year_of_Release'], dtype=torch.float32).unsqueeze(1)
print(x.shape)
print(x.dtype)

# create inputs df of all one-word columns
# do pd.dummies on the inputs df
# convert values to tensor
single_words = df[['Genre', 'Platform', 'Rating', 'Publisher', 'Developer']]
for column in single_words:
    one_hot_encoded = pd.get_dummies(single_words[column], dtype=float)
    t = torch.tensor(one_hot_encoded.values, dtype=torch.float32)
    print(t.shape)
    print(t.dtype)
    x = torch.cat((x, t), dim=1)
    print(x.shape)

# create inputs df of all float value columns
floats = df[['Critic_Score', 'User_Score', 'User_Count']]
for column in floats:
    column = pd.to_numeric(floats[column], errors='coerce')  # for any values aren't floats (surprise strings)
    column = column.fillna(column.mean())  # fill missing values
    column = column / 100  # standardize in 0-1 scale
    t = torch.tensor(column, dtype=torch.float32).unsqueeze(1)
    print(t.shape)
    print(t.dtype)
    x = torch.cat((x, t), dim=1)
    print('float_x')
    print(x.shape)

# Load pre-trained Word2Vec model using gensim
# Make sure to download the pre-trained model first
model_path = '/Users/skyler/Downloads/GoogleNews-vectors-negative300.bin'
# Load the model
word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)


def sentence_to_padded_embeddings(sentence, max_length=5):
    tokens = sentence.split()  # convert column to its tokens
    embeddings = []

    # Convert tokens to embeddings
    for token in tokens:
        if token in word_vectors:
            embeddings.append(word_vectors[token])
            # add the embedded version of the token to the list
        else:
            embeddings.append(np.zeros((300,)))  # if word not in vocab
            # what is the 300 for?
            # what does np.zeros do?

    # Pad sequences to the left to have a fixed length of max_length
    if len(embeddings) < max_length:
        padding = [np.zeros((300,)) for _ in range(max_length - len(embeddings))]
        # what is the 300 for?
        # list of 0's with the length of
        # the gap between current embeddings length and max length
        embeddings = padding + embeddings
    elif len(embeddings) > max_length:
        embeddings = embeddings[:max_length]
        # cuts the list off at max length

    return np.array(embeddings)

df['Name'] = df['Name'].astype(str)
df['padded_embeddings'] = df['Name'].apply(sentence_to_padded_embeddings)
x_embeddings = torch.tensor(df['padded_embeddings'])
print(x_embeddings)
print(x_embeddings.shape)
x = x.unsqueeze(2)
print(x.shape)

x = torch.cat((x, x_embeddings), dim=0)
y = torch.tensor(df["NA_Sales"] / 100, dtype=torch.float32).unsqueeze(1)

kf = KFold(n_splits=5, random_state=None)
cross_val = 0
input_size = x.shape[1]
print(input_size)

# does it make sense to combine so many variables? are they comparable?
for train_index, test_index in kf.split(x, y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = nn.Linear(input_size,
                      1)  # first is # inputs, second is # outputs #print(model) print(list(model.parameters()))
    # loss = nn.MSELoss()  # print(loss)
    loss = nn.MAELoss() #this isn't squared, more interpretable result
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)  # print(optimizer)

    train_loss_list = []
    test_loss_list = []
    for epoch in range(1000):  # train the model
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
        cross_val += epoch_loss  # smallest loss

    import matplotlib.pyplot as plt
    plt.plot(train_loss_list,  label = 'Train Loss')
    plt.plot(test_loss_list, label= 'Test Loss')
    plt.legend()
    plt.show()

print("Cross Validation Score: ", cross_val / 5)


# separate 80/20 train ans test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)



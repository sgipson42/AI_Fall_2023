import torchvision
from torchvision import transforms
import torch
from torch import nn
import torchsummary

# test these:
# wandb for tracking results
# sweep for trying different parameters
# dropout -- neuron node elimination, eliminating noise

def main(): # what is this code actually predicting?
    img_dimensions = 224
    batch_size = 32  # feed them in all at once

    # Define the transformations to be applied to the images
    img_transforms = transforms.Compose([
        transforms.Resize((img_dimensions, img_dimensions)),  # just resizes all images to dimensions you want
        transforms.ToTensor()  # always end the transformations with this --why?
    ])

    # Load the dataset
    train_dataset = torchvision.datasets.ImageFolder(
        root='/Users/skyler/Downloads/train',
        # to the training folder # there are subfolders in this it will go through
        transform=img_transforms)

    # Split the dataset into training and validation sets
    # Validation is test dataset?
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [20000, 5000])

    # Create training and validation dataloaders
    # Loads the 32 images
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True
    )

    validation_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True
    )

    model = nn.Sequential(  # if this can recognize cats vs. dogs, its happening here
        # could do convolution convolution then a max pool, could have more linear layers, etc
        nn.Conv2d(3, 16, kernel_size=3, padding=1), #input channels, # output channels, kernel_size
        # 3 inchannels, 16 out channels, kind of like a feed forward but is retaining 2D shape
        # 3 in channels are RGB, and each is fed to 16 channels
        # kernel size is 3 by 3, this is being run over each part of the image and producing a new value (the 3x3 kernel has 9 different alues, its training to learn these 9 values)
        # stride default is 1
        # padding 1 with a 3 by 3 is right to do the corners correctly
        # 224 for the 16 channels
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # kernel size then stride size (reducing height and width both by 2)
        # does the kernel thing again where it says of all the values in the kernel, keep the largest and it will be the center of the kernel
        # will halve it on both sides, so its 1/4 of the size
        # now they are 112 * 112
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        # now 56*56
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        #now 28*28
        nn.Flatten(),  # no longer 2D, goes to 1D (no longer concerned about what the neighbor of what is)
        nn.Linear(64 * 28 * 28, 512),  # have to tell it the # of inputs, # of outputs (the sizes, for pixels)
        # 64 is # of channels, then dimensions
        nn.ReLU(), # need this between two linear layers
        nn.Linear(512, 2) # get 2 outputs (y_hat)

    )

    # model = model.to("mps:0") # this sends the data to the GPU, instead of staying on the CPU
    torchsummary.summary(model, (3, img_dimensions, img_dimensions))
    # will tell you the dimensions of everything -- the first dimension that comes out is the batch size
    # the parameters are the kernels and the biases from the 3 channels
    # relu doesnt learn any parameters, and neither does maxpool(just walks across image and takes max pixel)
    # from flatten to linear the parameters are the number of weights and biases

    # now we have to train it
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        for batch in train_dataloader:
            x, y = batch  # specific to image dataloader # what are x and y here?
            # x = x.to("mps:0")
            print('x')
            print(x)
            print('y')
            print(y)
            # y = y.to("mps:0")
            y_hat = model(x)
            print(y_hat) # tensor of 32*2--2 outputs per image, 32 images
            loss = loss_fn(y_hat, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f'Epoch {epoch} loss: {loss}')
        correct = 0
        total = 0
        for batch in validation_dataloader:
            x, y = batch
            y_hat = model(x)
            _, predicted = torch.max(y_hat.data, 1)  # what?
            total += y.size(0)  # adding what?
            correct += (predicted == y).sum().item()  # adding what?
        print(
            f'Epoch {epoch} accuracy: {correct / total}')  # is this an actual way to do accuracy? # dont need to do cross val for images?


if __name__ == '__main__':
    main()

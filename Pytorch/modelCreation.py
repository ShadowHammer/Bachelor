import torch
import torch.nn as  nn
import torch.optim as  optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import os

import matplotlib.pyplot as plt

import time
from datetime import timedelta
import numpy as np

absolute_path = os.path.dirname(__file__)
relative_train_path, relative_valid_path, relative_test_path = ("archive",
                                                                "archive",
                                                                "archive")
train_path, valid_path, test_path = (os.path.join(absolute_path, relative_train_path),
                                     os.path.join(absolute_path, relative_valid_path),
                                     os.path.join(absolute_path, relative_test_path))

"""
In average the size of normal photos are 1669*1380 (Max: 2890*2663, Min:624*747).
In average the size of pneumonia photos are 1193*827 (Max: ,Min:).
We want the size of the images to be quadratic, because some pictures have longest dimensions in the width and some not.
As a conclusion to this we resize the images to 1024*1024.
"""
# the training transforms
train_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 5)),
    transforms.RandomRotation(degrees=(-90, 90)),
    transforms.ToTensor()
])
# the validation transforms
valid_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor()
])
# the test transforms
test_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor()
])

batch_size = 64


"""
ImageFolder creates a costume dataset class for us
"""
# training dataset
train_dataset = datasets.ImageFolder(
    root=train_path,
    transform=train_transform
)
# validation dataset
valid_dataset = datasets.ImageFolder(
    root=valid_path,
    transform=valid_transform
)
# test dataset
test_dataset = datasets.ImageFolder(
    root=test_path,
    transform=test_transform
)


# training data loaders
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
# validation data loaders
valid_loader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False
)
# test data loaders
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)




"""
print(f'Length of dataset is {len(train_dataset)}, shape of data {train_dataset[0][0].shape}')

# Run this to test your data loader
images, labels = next(iter(train_loader))
# helper.imshow(images[0], normalize=False)
plt.imshow(images[0].T)
plt.show()
"""
print(f'{torch.version.cuda}')
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
print(f'Using {device} device')


class FinalNetwork(nn.Module):
    def __init__(self):
        super(FinalNetwork,self).__init__()

        # Branch 1
        self.conv11 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=3) # 1024 --> 340
        self.maxpool11 = nn.MaxPool2d(kernel_size=2) # 340 --> 170
        self.conv12 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3) # 170 --> 168
        self.maxpool12 = nn.MaxPool2d(kernel_size=2) # 168 --> 84
        self.conv13 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3) # 84 --> 82
        self.maxpool13 = nn.MaxPool2d(kernel_size=2) # 82 --> 41
        self.conv14 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1) # 41 --> 40
        self.maxpool14 = nn.MaxPool2d(kernel_size=2) # 40 --> 20
        self.conv15 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3) # 20 --> 18

        # Classifier
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_features=4*18*18, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=2)

    def forward(self,x):
        x = self.relu(self.conv11(x))
        x = self.maxpool11(x)
        x = self.relu(self.conv12(x))
        x = self.maxpool12(x)
        x = self.relu(self.conv13(x))
        x = self.maxpool13(x)
        x = self.relu(self.conv14(x))
        x = self.maxpool14(x)
        x = self.relu(self.conv15(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


"""
This is the loop for training the model. Here the training dataset will be loaded in as the dataloader.
The optimizer chosen is SGD 
As loss function Cross Entropy Loss has been chosen as this has been the main function of discussion in class, it gives a good performance, and is often used for binary classification.
Before entering the training loop the model is set to train() so all gradients are calculated to learn from.
Gradients are reset when entering with a new batch to get a new starting point. Then the model predicts from the given data.
The loss is then calculated with the previously mentioned loss function afterwards the gradients are the calculated with backward.
With the gradients the weights are then updated with the optimizer.
"""
def train(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0
    for data, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(data.to(device))
        loss = criterion(outputs, targets.to(device))
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    return running_loss / len(dataloader)

"""
This is the validation loop where the testing data is loaded to test the current model after training.
The model is set to evaluate before entering the loop so gradients which are not needed will not be calculated.
In the testing loop the loss is again calculated with the Cross Entropy Loss function which is summerized to get the average loss for the epoch.
The accuracy is calculated with the total amount of targets and the total amount of correct predictions.
"""
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0
    with torch.no_grad():
        for data, targets in dataloader:
            outputs = model(data.to(device))
            loss = criterion(outputs, targets.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += targets.to(device).size(0)
            correct += (predicted == targets.to(device)).sum().item()
            accuracy = 100 * correct / total
            print(accuracy)
            running_loss += loss.item()

    return [running_loss / len(dataloader), accuracy]

lr = 0.00000001
loss_data = []
evaluation_data = []

model = FinalNetwork().to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr, momentum=0.5)

num_epochs = 30
best_loss = float('inf')
epochs_without_improvement = 0
patience = 10

start = time.time()
for epoch in range(num_epochs):
    loss_data.append(train(model, train_loader, optimizer, criterion))
    evaluation_data.append(evaluate(model, test_loader))

    print(f"Epoch [{epoch+1}/{num_epochs}], Time stamp: {timedelta(seconds=(time.time()-start))}, Train loss: {loss_data[epoch]:.4f}, Validation loss: {evaluation_data[epoch][0]:.4f}, Current Accuracy: {evaluation_data[epoch][1]:.2f}")

    if evaluation_data[epoch][0] < best_loss:
        best_loss = evaluation_data[epoch][0]
        epochs_without_improvement = 0
        torch.save(model.state_dict(),absolute_path)
        print("Best model saved")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Early stop triggered")
        break

# Step 6: Plot the loss curve
#plt.figure(figsize=(num_epochs, 5))
plt.plot(loss_data, color='g', label='Train Loss')
plt.plot(np.array(evaluation_data)[:,0], color="c", label="Validation loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Curve')
plt.show()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import timedelta
import lightning as L
import draw_bb as bb


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
batch_size = 64
save_path = "save"
lr=0.001
resize = 244


train_transforms = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean),torch.Tensor(std))
])

test_transforms = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean),torch.Tensor(std))
])
valid_transforms = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.ToTensor()
])
print("Transform done")

# actual dataset
data_path = "data"

# Function to show transformed images
def show_transformed_images(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True)
    batch = next(iter(loader))
    images, labels = batch

    grid = torchvision.utils.make_grid(images, nrow=3)
    plt.figure(figsize=(11,11))
    plt.imshow(np.transpose(grid, (1,2,0)))
    print('labels:', labels)
    plt.show()

# Import data and apply transformations
test_train_data = torchvision.datasets.ImageFolder(root=data_path, transform=valid_transforms)
train_data = torchvision.datasets.ImageFolder(root=data_path, transform=train_transforms)
test_data = datasets.ImageFolder(root=data_path, transform=test_transforms)
valid_data = datasets.ImageFolder(root=data_path, transform=valid_transforms)

#print("Test data before transform: ")
#show_transformed_images(test_train_data)


# Create data loaders shuffle = True for training data
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

# Show transformed images
#print("Test data after tradnsform: ")
#show_transformed_images(train_data)

# use cuda if available
print(f'{torch.version.cuda}')
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
print(f'Using {device} device')

#device = 'cpu'

#Til hvis man løber tør for memory
#PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:256'


# implement resnet 50 model pretrained on the imagenet dataset
#resnet50_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# implement resnet 50 model not pretrained on the imagenet dataset
resnet50_model = models.resnet50()
# number of input features in the last layer
num_ftrs = resnet50_model.fc.in_features
# number of classes in the dataset
number_of_classes = 3 # maybe only 3 classes
# function that replaces the last layer with a new layer with the number of classes in the dataset
resnet50_model.fc = nn.Linear(num_ftrs, number_of_classes)
resnet50_model.to(device)
# widely used loss function for classification problems, good for multi-class problems as it provides a greater penalty for incorrect classifications
loss_function = nn.CrossEntropyLoss()

# optimizer that uses the stochastic gradient descent algorithm
# weight decay can be used to avoid overfitting
optimizer = optim.SGD(resnet50_model.parameters(), lr, momentum=0.9)

# --------------------------------------------------------------------------------------------------------------------- #
bb1 = bb.BoundingBox('pothole_xml/train')
bb1.run()
# --------------------------------------------------------------------------------------------------------------------- #
# Function to train the model
def train(model, train_loader, test_loader, optimizer, criterion, epochs):
    
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        i = 0

        for data in train_loader:
            images, labels = data # get the inputs and labels from the data loader

            images, labels = images.to(device), labels.to(device) # move the data to the device
            total += labels.size(0) # add the number of labels to the total

            optimizer.zero_grad() # zero the gradients to avoid accumulation

            outputs = model(images) # forward pass

            _, predicted = torch.max(outputs.data, 1) # get the predicted class from the model 

            loss = criterion(outputs, labels) # calculate the loss 
            
            loss.backward() # backpropagation to calculate the gradients

            optimizer.step() # update the weights based on the gradients calculated during the backward pass

            running_loss += loss.item() # add the loss to the running loss variable to calculate the average loss for the epoch
            running_correct += (predicted == labels).sum().item() # add the number of correct classifications to the running correct
            i += 1
            print("Iteration: ", i)
            

        epoch_loss = running_loss / len(train_loader) # calculate the loss for the epoch
        epoch_acc = 100 * running_correct / total # calculate the accuracy for the epoch
        print(" - Training dataset. got %d out of %d correct (%.2f%%). Epoch loss: %.2f" % (running_correct, total, epoch_acc, epoch_loss))
        evaluate_model_on_test_set(model, test_loader) # evaluate the model on the test set

    return model

def evaluate_model_on_test_set(model, test_loader):
    model.eval()
    predicted_correctly_on_epoch = 0
    total = 0
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            total += labels.size(0)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted_correctly_on_epoch += (predicted == labels).sum().item()
    epoch_acc = 100 * predicted_correctly_on_epoch / total
    print("Accuracy on test set: ", epoch_acc)


num_of_epochs = 10
loss_data = []
evaluation_data = []

model = resnet50_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)

#train(model, test_loader, test_loader, optimizer, criterion, num_of_epochs)
train(model, train_loader, test_loader, optimizer, criterion, num_of_epochs)


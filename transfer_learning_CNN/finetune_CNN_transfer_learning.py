import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torchvision 
from torchvision.datasets import CIFAR10
from torchvision.models import vgg16
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
# for show progress during training
from tqdm import tqdm
dirc= r'C:\Users\bansa\Documents\ML'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Transform Images
transformer=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
#load dataset
ds=CIFAR10(dirc,train=True,transform=transformer,download=True)

#dataloader
dl=DataLoader(ds,batch_size=32)

#Load VGG19 model
model= vgg16(pretrained=(True))
model.train()


for param in model.parameters():
    param.requires_grad = True


for name, param in model.named_parameters():
    if param.requires_grad:
        print( name)

#Freeze the upper layer and train only deep layer
for name, param in model.named_parameters():
    if name.split('.')[0]=='classifier':
        param.requires_grad=True

    else:
        param.requires_grad=False

# change the Output layer 
features=model.classifier[6].in_features
model.classifier[6]=nn.Linear(features,10)

#summary of model
summary(model, (3, 64, 64))

model.to(device)
#initialize Loss
loss_fn = nn.CrossEntropyLoss()
#Initialize Optimizer
optimizer= optim.Adam(model.parameters())

def finetune(epochs,dataloader,model,loss_fn,optimizer,transformer):
    for  epoch in range(epochs):
        print(epoch)
        
        loop=tqdm(enumerate(dataloader),leave=False,total=len(dataloader))
        for batch, (img,label) in loop:
            
            output=model(img)

            loss=loss_fn(output,label)
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            
            # Show progress while training
            loop.set_description(f'Epoch={epoch}/{epochs}')
            loop.set_postfix(loss=loss.item(),acc=torch.rand(1).item())

    return model

#Define function for check accuracies
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        
        loop=tqdm(enumerate(loader),leave=False,total=len(loader))
        for x, y in loop:
            
            x = x.to(device=device)
            y = y.to(device=device)

            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}') 
    
    model.train()

model=finetune(1, dataloader=dl, model=model, loss_fn=loss_fn, optimizer=optimizer, transformer=transformer)
check_accuracy(dl,model)
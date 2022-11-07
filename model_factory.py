################################################################################
# CSE 151B: Programming Assignment 3
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin, Yash, James
# Fall 2022
################################################################################
import torch
from torchvision.models import resnet50
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    '''
    A Custom CNN (Task 1) implemented using PyTorch modules based on the architecture in the PA writeup. 
    This will serve as the encoder for our Image Captioning problem.
    '''
    def __init__(self, outputs):
        '''
        Define the layers (convolutional, batchnorm, maxpool, fully connected, etc.)
        with the correct arguments
        
        Parameters:
            outputs => the number of output classes that the final fully connected layer
                       should map its input to
        '''
        super(CustomCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(out_channels= 64, kernel_size= 11, stride=1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.batchnorm = nn.BatchNorm2d(num_features)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2= Conv2d(out_channels = 128, kernel_size=5, padding=2)
        self.conv3 = Conv2d(out_channels = 256, kernel_size=3, padding=1)
        self.conv4 = Conv2d(out_channels = 256, kernel_size=3, padding=1)
        self.conv5 = Conv2d(out_channels = 128, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)    
        self.avgpool = AdaptiveAvgPool2d(kernel_size = 1)
        
        self.fc1 = nn.Linear(out_features = 1024)
        self.fc2 = Linear(out_features = 1024)
        self.fc3 = Linear(out_features = 300)


    def forward(self, x):
        '''
        Pass the input through each layer defined in the __init__() function
        in order.

        Parameters:
            x => Input to the CNN
        '''
        
        x = F.relu(self.norm(self.conv1(x)))
        x = self.maxpool1(x)
        x = F.relu(self.norm(self.conv2(x)))
        x = self.maxpool2(x)
        x = F.relu(self.norm(self.conv3(x)))
        x = F.relu(self.norm(self.conv4(x)))
        x = F.relu(self.norm(self.conv5(x)))
        x = self.maxpool2(x)
        x = self.avgpool(x)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return  self.fc3(x)


class CNN_LSTM(nn.Module):
    '''
    An encoder decoder architecture.
    Contains a reference to the CNN encoder based on model_type config value.
    Contains an LSTM implemented using PyTorch modules. This will serve as the decoder for our Image Captioning problem.
    '''
    def __init__(self, config_data, vocab):
        '''
        Initialize the embedding layer, LSTM, and anything else you might need.
        '''
        super(CNN_LSTM, self).__init__()
        self.vocab = vocab
        self.hidden_size = config_data['model']['hidden_size']
        self.embedding_size = config_data['model']['embedding_size']
        self.model_type = config_data['model']['model_type']
        self.max_length = config_data['generation']['max_length']
        self.deterministic = config_data['generation']['deterministic']
        self.temp = config_data['generation']['temperature']

        self.embed = nn.Embedding(self.vocab, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, num_layers, batch_first=True)



    def forward(self, images, captions, teacher_forcing=False):
        '''
        Forward function for this model.
        If teacher forcing is true:
            - Pass encoded images to the LSTM at first time step.
            - Pass each encoded caption one word at a time to the LSTM at every time step after the first one.
        Else:
            - Pass encoded images to the LSTM at first time step.
            - Pass output from previous time step through the LSTM at subsequent time steps
            - Generate predicted caption from the output based on whether we are generating them deterministically or not.
        '''
             '''
        x = F.relu(self.norm(self.conv1(x)))
        x = self.maxpool1(x)
        x = F.relu(self.norm(self.conv2(x)))
        x = self.maxpool2(x)
        x = F.relu(self.norm(self.conv3(x)))
        x = F.relu(self.norm(self.conv4(x)))
        x = F.relu(self.norm(self.conv5(x)))
        x = self.maxpool2(x)
        x = self.avgpool(x)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return  self.fc3(x)


def get_model(config_data, vocab):
    '''
    Return the LSTM model
    '''
    return CNN_LSTM(config_data, vocab)

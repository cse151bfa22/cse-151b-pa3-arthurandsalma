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
        
        self.conv1 = nn.Conv2d(out_channels=64, kernel_size=11, stride=1, in_channels=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.batchnorm = nn.BatchNorm2d(outputs)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(out_channels=128, kernel_size=5, padding=2, in_channels=64)
        self.conv3 = nn.Conv2d(out_channels=256, kernel_size=3, padding=1, in_channels=128)
        self.conv4 = nn.Conv2d(out_channels=256, kernel_size=3, padding=1, in_channels=256)
        self.conv5 = nn.Conv2d(out_channels=128, kernel_size=3, padding=1, in_channels=256)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)    
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        
        self.fc1 = nn.Linear(in_features=128, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=outputs)


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
        
        return self.fc3(x)


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

        self.embed = nn.Embedding(len(self.vocab), self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, 512, batch_first=True, proj_size=self.embedding_size)
        self.softmax = nn.Softmax(self.embedding_size)


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
        def out_to_word(r):
            p = torch.rand(1)[0]
            i = 0
            while True:
                if p < r[i]:
                    return self.vocab.idx2word[i]
                p -= r[i]
                i += 1

        out, (h, c) = self.lstm(images)
        s = self.softmax(out)
        res = [out_to_word(s)]
        i = 0
        while True:
            wordidx = captions[0][len(res) - 1] if teacher_forcing else res[-1]
            input = self.embed(wordidx)
            out, (h, c) = self.lstm(input, (h, c))
            s = self.softmax(out)
            w = out_to_word(s)
            res.append(w)
            i += 1
            if w == "<end>" or i > 20:
                break

        return res

def get_model(config_data, vocab):
    '''
    Return the LSTM model
    '''
    return CNN_LSTM(config_data, vocab)

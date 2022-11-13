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
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)    
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        
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
        
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = self.maxpool1(x)
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = self.maxpool2(x)
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = F.relu(self.batchnorm3(self.conv4(x)))
        x = F.relu(self.batchnorm2(self.conv5(x)))
        x = self.maxpool3(x)
        x = torch.flatten(self.avgpool(x),1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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

        if self.model_type=="Resnet":
            self.cnn = resnet50(pretrained=True)
            self.cnn.fc = nn.Linear(2048, self.hidden_size)
        elif self.model_type=="Custom":
            self.cnn = CustomCNN(self.embedding_size)
        self.embed = nn.Embedding(len(self.vocab), self.embedding_size)
        self.lstm = nn.LSTM(input_size=self.embedding_size,
                        hidden_size=self.hidden_size,
                        num_layers=2,
                        batch_first=True)
                        #proj_size=self.hidden_size)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=len(self.vocab))
        self.softmax = nn.Softmax(2)


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
        if teacher_forcing:
            image_embeddings = self.cnn(images).unsqueeze(1)
            caption_emebddings = self.embed(captions[:, :-1]) # do the one with two columns
            # you might have to unsqueeze to flatten it - TORCH.UNSQUEEZE (prolly for image embedding)
            # print(f'Image embeddings: {image_embeddings.size()}')
            # print(f'Caption embeddings: {caption_emebddings.size()}')
            embedding = torch.cat((image_embeddings, caption_emebddings), dim=1)
            # print(f'Embedding size: {embedding.size()}')
            # then pass embeddings through lstm
            out, (h_n, c_n) = self.lstm(embedding)
            # then pass lstm outputs to fc layer
            out = self.fc(out) # I just removed self.softmax
            # return fc layer ouput, output is list of logits
            return out
        else: # if non-teacher forcing
            res = None # list of indices
            # (in non teacher forcing u won't concat)
            out = self.cnn(images)
            out = out.unsqueeze(1)
            h,c = None, None
            for i in range(self.max_length):
                # in each iteration, u want to get image embeddings
                # pass thru lstm
                if h is None:
                    out, (h, c) = self.lstm(out)
                else:
                    out, (h, c) = self.lstm(out, (h,c))
                # pass thru fully connected
                out = self.fc(out)
                # pass thru softmax
                # print(f'Output size at line 144: {out.size()}')
                # get argmax or torch multinomial sample
                if self.deterministic:
                    out = self.softmax(out, dim=2)
                    out = torch.argmax(out, dim=1)
                else:
                    temp_out = self.softmax(out / self.temp, dim=2)
                    idx = torch.multinomial(temp_out, num_samples=1)
                    out = out[idx]
                if i == 0:
                    res = out
                else:
                    res = torch.cat((res, out), dim=1)
                # that gives indeces
                # print(f'Output size at line 155: {out.size()}')
                # print(f'Res size at line 155: {res.size()}')
                out = self.embed(out)
                # pass index thru embedding layer, and that becomes input for next iter
            return res

def get_model(config_data, vocab):
    '''
    Return the LSTM model
    '''
    return CNN_LSTM(config_data, vocab)

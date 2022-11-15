################################################################################
# CSE 151B: Programming Assignment 3
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin, Yash, James
# Fall 2022
################################################################################

from enum import unique
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
import math
from tqdm import tqdm
from copy import deepcopy
from nltk.tokenize import word_tokenize
import caption_utils

ROOT_STATS_DIR = './experiment_data'
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model


# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco, self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)
        
        # Setup Experiment
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__early_stop = config_data['experiment']['early_stop']
        self.__patience = config_data['experiment']['patience']
        self.__batch_size = config_data['dataset']['batch_size']

        # Init Model
        self.__model = get_model(config_data, self.__vocab)
        self.__best_model = deepcopy(self.__model.state_dict())

        # criterion
        self.__criterion = torch.nn.CrossEntropyLoss()

        # optimizer
        optimizer = config_data['experiment']['optimizer']
        if optimizer == 'Adam':
            self.__optimizer = torch.optim.Adam(self.__model.parameters(), config_data['experiment']["learning_rate"])
        elif optimizer == 'SGD':
            self.__optimizer = torch.optim.SGD(self.__model.parameters(), config_data['experiment']["learning_rate"])


        # LR Scheduler
        lr_scheduler = config_data['experiment']['lr_scheduler']# TODO
        if lr_scheduler == 'steplr':
            self.__lr_scheduler =  torch.optim.lr_scheduler.StepLR(self.__optimizer, config_data['experiment']["learning_rate"]) # TODO
        
        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()


    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])
        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        patience_count = 0
        min_loss = 100
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            print(f'Epoch {epoch + 1}')
            print('--------')
            start_time = datetime.now()
            self.__current_epoch = epoch
            print('Training...')
            print('-----------')
            train_loss = self.__train()
            print('Validating...')
            print('-------------')
            val_loss = self.__val()

            # save best model
            if val_loss < min_loss:
                min_loss = val_loss
                self.__best_model = deepcopy(self.__model.state_dict())

            # early stop if model starts overfitting
            if self.__early_stop:
                if epoch > 0 and val_loss > self.__val_losses[epoch - 1]:
                    patience_count += 1
                if patience_count >= self.__patience:
                    print('\nEarly stopping!')
                    break

            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()
            if self.__lr_scheduler is not None:
                self.__lr_scheduler.step()
        self.__model.load_state_dict(self.__best_model)

    def __compute_loss(self, images, captions):
        """
        Computes the loss after a forward pass through the model
        Forward pass is performed within the model
        """
        output = self.__model(images, captions, teacher_forcing=True)
        output = torch.transpose(output, 1,2)
        return self.__criterion(output, captions)

    def __train(self):
        """
        Trains the model for one epoch using teacher forcing and minibatch stochastic gradient descent
        """
        run_loss = 0
        self.__model.train()
        for i, data in enumerate(self.__train_loader):
            images, labels, image_IDs = data
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            self.__optimizer.zero_grad()
            
            loss = self.__compute_loss(images, labels)
            loss.backward()

            self.__optimizer.step()

            run_loss += loss.item()

            if i % 100==1:
                run_avg_loss = run_loss / (i)
                print(f'Avg loss at batch {i}: {run_avg_loss}')
                
        
        train_loss = run_loss / len(self.__train_loader)
        return train_loss

    def __generate_captions(self, img_id, outputs, testing):
        """
        Generate captions without teacher forcing
        Params:
            img_id: Image Id for which caption is being generated
            outputs: output from the forward pass for this img_id
            testing: whether the image_id comes from the validation or test set
        Returns:
            tuple (list of original captions, predicted caption)
        """
        captionDict = None
        if testing:
            captionDict  = self.__coco_test.imgToAnns[img_id]
            captionDict = [word_tokenize(caption['caption'].lower()) for caption in captionDict]
        else:
            captionDict = self.__coco.imgToAnns[img_id]
            captionDict = [word_tokenize(caption['caption'].lower()) for caption in captionDict]

        pred = []
        for output in outputs:
            pred.append(self.__vocab.idx2word[output])
        print(self.__str_captions(img_id, captionDict,pred))
        return (captionDict, pred)

    def __str_captions(self, img_id, original_captions, predicted_caption):
        """
            !OPTIONAL UTILITY FUNCTION!
            Create a string for logging ground truth and predicted captions for given img_id
        """
        result_str = "Captions: Img ID: {},\nActual: {},\nPredicted: {}\n".format(
            img_id, original_captions, predicted_caption)
        return result_str

    def __val(self):
        """
        Validate the model for one epoch using teacher forcing
        """
        run_loss = 0
        with torch.no_grad():
            for i, data in enumerate(self.__val_loader):
                images, labels, image_IDs = data
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()
                outputs = self.__model(images, labels, teacher_forcing=False)
                loss = self.__compute_loss(images, labels, outputs)

                run_loss += loss.item()
                outputs = outputs.cpu()
                captionDict, pred = self.__generate_captions(image_IDs[0], outputs[0], testing=False)
                captionDict, pred = self.__generate_captions(image_IDs[15], outputs[15], testing=False)
                captionDict, pred = self.__generate_captions(image_IDs[19], outputs[19], testing=False)
                if i % 100==1:
                    run_avg_loss = run_loss / (i)
                    print(f'Avg loss at batch {i}: {run_avg_loss}')
                
            
            val_loss = run_loss / len(self.__val_loader)
        return val_loss

    def test(self):
        """
        Test the best model on test data. Generate captions and calculate bleu scores
        """
        def filterTokens(pred):
            tokens = ['<start>', '<end>', '<pad>']
            if pred in tokens:
                return False
            else:
                return True
        run_loss = 0
        bleu1, bleu4, captions = [], [], []
        with torch.no_grad():
            uniqueImageIDs = set()
            for i, data in enumerate(self.__test_loader):
                images, labels, image_IDs = data
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()
                loss = self.__compute_loss(images, labels)

                run_loss += loss.item()

                if i % 200 == 1:
                    run_avg_loss = run_loss / (i)
                    print(run_avg_loss)
                    
                outputs = self.__model(images, labels, teacher_forcing = False)
                outputs = outputs.cpu()
                for idx in range(len(image_IDs)):
                    if image_IDs[idx] not in uniqueImageIDs:
                        captionDict, pred = self.__generate_captions(image_IDs[idx], outputs[idx], testing=True)
                        pred = list(filter(filterTokens, pred))
                        captions.append(self.__str_captions(image_IDs[idx], captionDict,pred))
                        bleu1.append(caption_utils.bleu1(captionDict,pred))
                        bleu4.append(caption_utils.bleu4(captionDict,pred))
                        uniqueImageIDs.add(image_IDs[idx])
        write_to_file_in_dir(self.__experiment_dir, 'bleu1.txt', bleu1)
        write_to_file_in_dir(self.__experiment_dir, 'bleu4.txt', bleu4)
        write_to_file_in_dir(self.__experiment_dir, 'testcaptions.txt', captions)
        plt.figure()
        plt.hist(bleu1)
        plt.xlabel("Bleu-1 Score")
        plt.ylabel("Number of examples")
        plt.savefig(os.path.join(self.__experiment_dir, "bleu1_hist.png"))
        plt.show()
        plt.figure()
        plt.hist(bleu4)
        plt.xlabel("Bleu-4 Score")
        plt.ylabel("Number of examples")
        plt.savefig(os.path.join(self.__experiment_dir, "bleu4_hist.png"))
        plt.show()
        test_loss = run_loss / len(self.__test_loader)
        test_stats = [test_loss, sum(bleu1)/len(bleu1), sum(bleu4)/len(bleu4)]
        write_to_file_in_dir(self.__experiment_dir, 'test_stats.txt', test_stats)
        print(f'Avg Test Loss: {test_loss}')

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()

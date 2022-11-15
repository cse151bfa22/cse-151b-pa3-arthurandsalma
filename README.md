# PA 3: Image Captioning


## Contributors

<TODO>

## Task

  Our goal is to build an image captioning model using the Pytorch deep learning library. Image captioning consist of inputting an image to a model 
that outputs a caption summarising the content of the image. Image captioning models consist of 2 main components: a Convolutional Neural Network encoder and a long short-term memory decoder. For this work, we are using the Common Objects in Context (COCO)2015 Image Captioning task dataset, be trained to  decoder the caption using a mechanism called Teacher Forcing. 

## How to run
<TODO>


## Usage

* Define the configuration for your experiment. See `task-1-default-config.json` to see the structure and available options. You are free to modify and restructure the configuration as per your needs.
* Implement factories to return project specific models, datasets based on config. Add more flags as per requirement in the config.
* Implement `experiment.py` based on the project requirements.
* After defining the configuration (say `my_exp.json`) - simply run `python3 main.py my_exp` to start the experiment
* The logs, stats, plots and saved models would be stored in `./experiment_data/my_exp` dir.
* To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training or evaluate performance.

## Files
- `main.py`: Main driver class
- `experiment.py`: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments.
- `dataset_factory.py`: Factory to build datasets based on config
- `model_factory.py`: Factory to build models based on config
- `file_utils.py`: utility functions for handling files
- `caption_utils.py`: utility functions to generate bleu scores
- `vocab.py`: A simple Vocabulary wrapper
- `coco_dataset.py`: A simple implementation of `torch.utils.data.Dataset` the Coco Dataset
- `get_datasets.ipynb`: A helper notebook to set up the dataset in your workspace

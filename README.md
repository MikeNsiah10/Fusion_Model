## Lightweight Parameterised Fusion Model

# Project Overview

This repository is designed for working with the MMF2F dataset, working with smaller fused text and audio distilled models,with unquie pooling strategies to predict turns , bakchannel and continue or keep speech . It includes scripts for data handling, training, evaluation, and example initialsation of the trained models. The code leverages the pytorch for these implementations.
All the implementation of this project was done on the hpc of my university(University of Osnabrueck).All .sh diles were used for job execution on the hpc

## Repository Structure
```
/Data Preprocessing                # Contains scripts for context tokenization and audio segment extraction,downloading audio and removing files with no audio
/Data/preprocess                   # Contains the training, testing and validation utterance level files of MMF2F datasets
    ├── train.csv       
    ├── test.csv                   
    ├── val.csv           
/Evaluation /results_plots                      #Contain metrics plots such as per class f1, macro f1 and balanced accuracy of the trained models
    ├─ eval.py          #scripts for evaluation of the models
    ├─ metrics.py       #scripts for inference and plotting for the test set
    ├─ loss_curve.py    #scripts for plotting train and val loss

/Example Usage
    ├─ example.py                 # exmaple script on how to instantiate the model for infernece
/Modules                          # Modules contain reusable components for training and evaluation
    ├── audio_pooling.py          # Script for implemetning different audio pooling strategies
    ├── text_pooling.py           # Script for implementing text pooling 
    ├── focal_loss.py             # loss function used for training
    └── context_collate.py        # for dataloader collation for training and evaluation
    └── weight_sampler.py         # for weighted sampling
/checkpoints                      # contain the saved checkpoints for each model
/csv files                        # contain final training, testing and validation set
/model weights                    # contain the weights of the models
/models                           # contains the models used 
    ├── distilhub.py                   # distilhubert
    └── text_head_1.py                 #distilgpt2
    ├─ fusion_with_modality.py          # fusion head
/teacher_trp_script                  
    ├─ teacher_model.py             # contain a modified version of TurnGPT         
requirements.txt                   # file with the different libraries used on the project

Note:Training was done on the HPC.
```
## Requirements

To use this repository, follow these steps:

1. Clone the Repository:
   ```
      git clone https://github.com/MikeNsiah10/Fusion_Model.git
   cd Modelling_project
   ```

2. Setting Up a Python Environment
It is recommended to use a virtual environment to manage the dependencies for this project. A virtual environment helps to isolate your project's dependencies from your global Python environment, avoiding potential conflicts.
```
   # Create a virtual environment in a directory named 'env'
   python3 -m venv env

    # Activate the virtual environment
    # On Windows
    env\Scripts\activate
    # On macOS/Linux
    source env/bin/activate
```
   
3. Install Dependencies:
   Make sure you have the necessary libraries installed. You can use pip to install them:
   ```
       pip install -r requirements.txt
   ```
   
4.Instantiate the trained models as used in the example folder.
Note:Trainign was done using contextualised training examples.(two previous utterances + current utterance)

## Purpose
This code is implemented to fuse two fintuned distilled text and audio models to predict turns, backchanel and continue speech.We trained and tested the model using contextualised examples of Multimodal face to face datasets.
Assesment of the model was done using per class f1, macro f1 score and balanced accuracy which are key metrics for evaluating dialogue models.

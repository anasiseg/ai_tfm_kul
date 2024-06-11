from src.trainingClass import trainingClass
from src.autoencoderModels.vaeModel import vaework12
from src.autoencoderModels.discriminator import discriminator
from enum import Enum
from src.stateReader import stateReader
from src.datasetManager import datasetManager

import matplotlib.pyplot as plt
import tkinter
import matplotlib
matplotlib.use('TkAgg')

data_path = 'images_2'
reader = stateReader()
setsManaget = datasetManager()

trainpctg = 0.9
options_data= Enum('OptionsData', ['DATA_AUTOENCODER_TRAINING', 'DATA_SHARP_EVOLUTION'])
OPTION_DATA_SELECTED = options_data.DATA_SHARP_EVOLUTION

if OPTION_DATA_SELECTED==options_data.DATA_AUTOENCODER_TRAINING:
    (trainset, testset) = setsManaget.preparing_data(data_path, trainpctg)
    prediction_dataset = [trainset]

elif OPTION_DATA_SELECTED==options_data.DATA_SHARP_EVOLUTION:
    prediction_dataset = setsManaget.loading_sharp_data(trainpctg)
    import torch
    testset = prediction_dataset[53]#torch.utils.data.ConcatDataset(prediction_dataset)

plt.imshow(testset.datasets[0].img_list[10])

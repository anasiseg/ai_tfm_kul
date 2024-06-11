from src.trainingClass import trainingClass
from src.autoencoderModels.vaeModel import vaework12
from src.autoencoderModels.discriminator import discriminator
from enum import Enum
from src.stateReader import stateReader
from src.datasetManager import datasetManager

data_path = 'images'
state_path = 'state/state_08022024.torch'
out_state_path =''
learning_rate=1e-3
weight_decay = 1e-6

#initialization of classes
trainingC = trainingClass(vaework12(8), learning_rate, tqdm=True)
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
    testset = torch.utils.data.ConcatDataset(prediction_dataset)

epochs = 50
valpctg = 0.9
k_cross = 0
batch_size = 128

options= Enum('Options', ['DISPLAY_LATENT_SPACE'])
OPTION_SELECTED = options.DISPLAY_LATENT_SPACE


if OPTION_SELECTED==options.DISPLAY_LATENT_SPACE:
    #Analysing the latent space
    print("Analysing the latent space")
    (L, V) = reader.readState(state_path, learning_rate, trainingC.getModel(), trainingC.getOptimizer())
    trainingC.plotTestSet(testset,batch_size)






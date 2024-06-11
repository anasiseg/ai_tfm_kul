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
trainingC = trainingClass(vaework12(8), learning_rate, tqdm=False)
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
    trainset = torch.utils.data.ConcatDataset(prediction_dataset)

learning_rate_dis=1e-4
weight_decay_dis = 1e-6
dis = discriminator()
trainingC.addingDiscriminatorModel(dis, learning_rate_dis, weight_decay_dis)

# trainingC.addAimRunning(2e-3, batch_size)


epochs = 50
valpctg = 0.9
k_cross = 0
batch_size = 128

options= Enum('Options', ['TRAINING_SCRATCH', 'TRAINING_PRELOADED', 'DISPLAY_RESULTS', 'TRAINING_PREDICTION', 'ANALYSIS_LATENT_SPACE'])
OPTION_SELECTED = options.TRAINING_PREDICTION


if OPTION_SELECTED==options.TRAINING_SCRATCH:
    (modeltrained, optimizer, L, V) = trainingC.trainPhase(epochs, trainset, valpctg, k_cross, batch_size)
    reader.saveState(L, V, out_state_path, modeltrained, optimizer)

elif OPTION_SELECTED==options.TRAINING_PRELOADED:
    (L, V) = reader.readState(state_path, learning_rate, trainingC.getModel(), trainingC.getOptimizer())
    (modeltrained, optimizer, L, V) = trainingC.trainPhase(epochs,  trainset, valpctg, k_cross, batch_size, L)
    reader.saveState(L, V, out_state_path, modeltrained, optimizer)

elif OPTION_SELECTED==options.DISPLAY_RESULTS:
    (L, V) = reader.readState(state_path, learning_rate, trainingC.getModel(), trainingC.getOptimizer())
    trainingC.plotTestSet(testset,batch_size)

elif OPTION_SELECTED==options.TRAINING_PREDICTION:
    (L, V) = reader.readState(state_path, learning_rate, trainingC.getModel(), trainingC.getOptimizer())
    trainingC.getembeddings(prediction_dataset, batch_size)

elif OPTION_SELECTED==options.ANALYSIS_LATENT_SPACE:
    #Analysing the latent space
    print("Analysing the latent space")






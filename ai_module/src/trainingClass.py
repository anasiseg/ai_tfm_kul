import torch
import matplotlib.pyplot as plt
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
# import the necessary packages
import os
from aim import Run, Image, Distribution
# from sklearn.model_selection import KFold

import time
from src.predict_model_training import predict_model_trainClass

class trainingClass():
    def __init__(self, model, learning_rate=0.1, tqdm=False):
        print("training class created ")
        self.DISCRIMINATOR=False
        self.AIM=False
        self.VALIDATION=False

        self.TQDM=tqdm

        
        self.beta=2.0

        self.model=model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        torch.cuda.empty_cache()
        torch.manual_seed(1)
        np.random.seed(1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        print("device is equal to", self.device)

    def getModel(self):
        return self.model
    
    def getOptimizer(self):
        return self.optimizer


    def addingDiscriminatorModel(self, dis, lr, weight_decay):
        self.dis = dis
        if torch.cuda.is_available():
            self.dis.cuda()

        self.lossdfct = torch.nn.BCELoss()
        self.optimD    = torch.optim.Adam(self.dis.parameters(), lr=lr, weight_decay=weight_decay)
        self.DISCRIMINATOR=True

    
    def addAimRunning(self, lr, batchsz):
        self.run = Run()
        self.run["hparams"] = {
                "learning_rate" : lr,
                "batch_size": batchsz,
                "batch_norm": True,
                "drop_out": False,
        }
        self.AIM=True
      

    def training_set(self, dataloader):

        L0=0
        if self.TQDM:
            loop = tqdm(dataloader)
        else:
            loop = dataloader

        for (image, _) in loop:
            # # Reshaping the image to (-1, 784)
            # image = image.reshape(-1, 3, 128*256)
            image = image.to(self.device)

            # Output of Autoencoder
            reconstructed, mu, lvar = self.model(image)
            
            # Calculating the loss function
            loss, _, _ = self.lossfct(reconstructed, image, mu, lvar, self.beta)

            if self.DISCRIMINATOR:
                yhat  = self.dis(reconstructed)
                lossD = self.lossdfct(yhat, torch.ones (len(image), 1, device=self.device))
                loss = loss + lossD

            L0 = L0 + loss.cpu().detach().numpy()
            
            # The gradients are set to zero,
            # the gradient is computed and stored.
            # .step() performs parameter update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.DISCRIMINATOR:
                self.optimD= self.computeDiscriminatorLoss(self.optimD, image)

        return (L0, image, reconstructed)


    def validating_set(self, dataset):
        V0 = 0
        for i, (x,_) in enumerate(dataset,0):
            x = x.to(self.device)
            y, mu, lvar = self.model(x)
            loss, bce, kld = self.lossfct(y, x,  mu, lvar, self.beta)
            V0 = V0 + loss.cpu().detach().numpy()
        return V0
    
    def training_model_epoch(self, epoch, train_loader, val_loader):
        self.model.to(self.device)
        self.model.train()
        
        (L0, image, reconstructed) = self.training_set(train_loader)

        if self.AIM:
            self.runningAim(L0, image[-1].cpu(), reconstructed[-1].cpu(), epoch)

        ## Validating
        #---------
        if self.VALIDATION:
            self.model.eval()
            V0 = self.validating_set(val_loader)
            V0 = V0/len(val_loader)
        else:
            V0=0
            
        
        return (L0/len(train_loader),V0)

    def early_stopping(self, train_loss, validation_loss, min_delta, tolerance):
        counter = 0
        if (validation_loss - train_loss)/train_loss > min_delta:
            counter +=1
            if counter >= tolerance:
                return True
        return False
        
    def KfoldsTrain(self, epoch, epochs, dataset, k_folds, batch_size):
        # # kf = KFold(n_splits=k_folds, shuffle=True)
        # L_fold=[]
        # V_fold=[]
        # for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        #     print(f"Fold {fold + 1}")
        #     print("-------")

        #     # Define the data loaders for the current fold
        #     train_loader = torch.utils.data.DataLoader(
        #         dataset=dataset,
        #         batch_size=batch_size,
        #         sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        #     )
        #     val_loader = torch.utils.data.DataLoader(
        #         dataset=dataset,
        #         batch_size=batch_size,
        #         sampler=torch.utils.data.SubsetRandomSampler(test_idx),
        #     )

        #     (L0, V0) = self.training_model_epoch(epoch, train_loader, val_loader)
        #     L_fold.append(L0)
        #     if self.VALIDATION:
        #         V_fold.append(V0)

        # return (np.mean(L_fold),np.mean(V_fold))
        print("")

    
    def trainPhase(self, epochs, dataset, validationpctg, k_folds, batch_size, L=[], V=[]):

        if validationpctg!=0:
            trainset, valset = torch.utils.data.random_split(dataset, [len(dataset)-int(validationpctg*len(dataset)), int(validationpctg*len(dataset))])
            self.VALIDATION=True


        for epoch in range(epochs):
            if k_folds!=0:
                self.VALIDATION=True
                (L0,V0) = self.KfoldsTrain(epoch, epochs, dataset, k_folds, batch_size)
            else:
                trainloader = torch.utils.data.DataLoader(
                    dataset = trainset,
                    batch_size = batch_size,
                    shuffle = True, drop_last=True)
                if self.VALIDATION:
                    validationloader = torch.utils.data.DataLoader(dataset = valset,
                                            batch_size = batch_size,
                                            shuffle = True, drop_last=True)
                (L0, V0) = self.training_model_epoch(epoch, trainloader, validationloader)

            L.append(L0)
            print(f"Epoch {epoch+1}/{epochs} , Train Loss: {L[-1]:.4f} ")
            if self.VALIDATION:
                V.append(V0)
                print(f"Epoch {epoch+1}/{epochs} , Validation Loss: {V[-1]:.4f} ")
            
            if self.early_stopping(L[-1], V[-1], min_delta=0.05, tolerance = 20):
                print("We are at epoch:", epoch)
                break

        return (self.model, self.optimizer, L, V)

    def plotTestSet(self, testset, batch_size):

        if len(testset)< batch_size:
            testloader = torch.utils.data.DataLoader(dataset = testset,
                                            batch_size = len(testset),
                                            shuffle = True, drop_last=True)
        else:
            testloader = torch.utils.data.DataLoader(dataset = testset,
                                            batch_size = batch_size,
                                            shuffle = True, drop_last=True)
        
        f, axarr = plt.subplots(1,2)
        T0=0
        firstImages=True
            
        if self.TQDM:
            loop = tqdm(testloader)
        else:
            loop = testloader

        for (x,_) in loop:
            # Reshape the array for plotting
            # item = item.reshape(-1, 128, 256)
            x = x.to(self.device)
            y, mu, lvar = self.model(x)
            # loss, bce, kld = self.lossfct(y, x, mu, lvar, self.beta)
            # T0 = T0 + loss.cpu().detach().numpy()
            if firstImages:
                axarr[0].imshow(x.cpu()[10].permute(1, 2, 0))
                axarr[1].imshow(y.cpu()[10].permute(1, 2, 0).detach().numpy())
                firstImages=False
                break

        print(f' Test loss: {T0/len(testloader):.4f}')
            
        
        plt.show()
        torch.cuda.empty_cache()
    
    def computeDiscriminatorLoss(self, optimD, x):
        ## Discriminator
        #---------------

        # Compute (O) loss
        yO = torch.ones (len(x), 1, device=self.device)
        yhatO = self.dis(x)
        LossO = self.lossdfct(yhatO, yO)

        # Compute (R) loss
        yR = torch.zeros(len(x), 1, device=self.device)
        yhatR,_,_ = self.model(x)
        yhatR = self.dis(yhatR)
        LossR = self.lossdfct(yhatR, yR)

        # Optimize discriminator
        LossD = LossO + LossR
        optimD.zero_grad()
        LossD.backward()
        optimD.step()
        return optimD

    def lossfct(self, y, x, mu, lvar, beta):
            BCE = F.binary_cross_entropy(y, x, reduction='sum') #size_average -> true average, false sum
            KLD = -0.5 *beta* torch.sum(1 + lvar - mu.pow(2) - lvar.exp())
            return  (BCE + KLD)/len(x), BCE, KLD
    
    def runningAim(self, loss, input, output, epoch):
        self.run.track(loss, name='loss', epoch=epoch, context={"subset":"train"})
        self.run.track(Image(input), name="the input" , epoch=epoch, context={"subset":"train"})
        self.run.track(Image(output), name="the output", epoch=epoch, context={"subset":"train"})


    def getembeddings(self, dataset_array, batch_size):
        latent_space_array=[]
        for dataset in dataset_array:
            if len(dataset)!=0:
                if len(dataset)< batch_size:
                    dataloader = torch.utils.data.DataLoader(dataset = dataset,
                                                    batch_size = len(dataset),
                                                    shuffle = True, drop_last=True)
                else:
                    dataloader = torch.utils.data.DataLoader(dataset = dataset,
                                                    batch_size = batch_size,
                                                    shuffle = True, drop_last=True)
                    
                if self.TQDM:
                    loop = tqdm(dataloader)
                else:
                    loop = dataloader

                array=None
                for (x,_) in loop:
                    # Reshape the array for plotting
                    # item = item.reshape(-1, 128, 256)
                    x = x.to(self.device)
                    y, mu, lvar = self.model(x)
                    latent_array = torch.cat((mu, lvar), 1)
                    if array is None:
                        array = latent_array
                    else:
                        array = torch.cat((array, latent_array), 0)
                
                latent_space_array.append(array)

        self.predicting_fit(latent_space_array, batch_size)
        
        torch.cuda.empty_cache()


    def predicting_fit(self, data_array, batch_size): 
        """
        Training Model, we use the fit() function 
        """
        from src.configurations.cfg_easytransformer import easyAttn_config as cfg
        X = None
        Y = None
        #convert data to correct size and to loader 
        for data in data_array:
            data = data.to("cpu").detach().numpy()
            predict_training = predict_model_trainClass()
            (x, y) = predict_training.make_Sequence(data,cfg.in_dim,cfg.next_step)
            if X is None and Y is None:
                X=x
                Y=y
            else:
                X = np.concatenate((X, x), axis=0)
                Y = np.concatenate((Y, y), axis=0)
        batch_size = 4
        dataset_pred = torch.utils.data.TensorDataset(torch.from_numpy(X),torch.from_numpy(Y))
        dLoader=  torch.utils.data.DataLoader(
                    dataset = dataset_pred,
                    batch_size = batch_size,
                    shuffle = True, drop_last=True)
        
        model = predict_training.get_model_prediction()
        (loss_fn, opt, opt_sch) = predict_training.compile()
        s_t = time.time()
        history = predict_training.fitting(self.device, 
                            model,
                            dLoader, 
                            loss_fn,
                            100,
                            opt,
                            None, 
                            scheduler=opt_sch)
        e_t = time.time()
        cost_time = e_t - s_t
        
        print(f"INFO: Training FINISH, Cost Time: {cost_time:.2f}s")
        
        check_point = { "model":self.model.state_dict(),
                        "history":history,
                        "time":cost_time}
        
        return check_point
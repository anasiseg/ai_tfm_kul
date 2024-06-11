from src.autoencoderModels.easytransformer import easyTransformerEncoder
import torch 
import numpy as np

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class predict_model_trainClass():
    def __init__(self):
        print("predicting model training class created ")
        
    def get_model_prediction(self):
        from src.configurations.cfg_easytransformer import easyAttn_config as cfg
        self.model = easyTransformerEncoder(d_input     = cfg.in_dim,
                                    d_output    = cfg.next_step,
                                    seqLen      = cfg.nmode,
                                    d_proj      = cfg.time_proj,
                                    d_model     = cfg.d_model,
                                    d_ff        = cfg.proj_dim,
                                    num_head    = cfg.num_head,
                                    num_layer   = cfg.num_block)
        return self.model
    
    def compile(self): 
        """
        Compile the model with optimizer, scheduler and loss function
        """
        loss_fn =   torch.nn.MSELoss()
        opt     =   torch.optim.Adam(self.model.parameters(),lr = 1e-3, eps=1e-7)
        opt_sch =  [  
                        torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma= (1 - 0.01)) 
                        ] 
        return (loss_fn, opt, opt_sch)
    
    def make_Sequence(self, data, in_dim, next_step):
        """
        Generate time-delay sequence data 

        Args: 
            cfg: A class contain the configuration of data 
            data: A numpy array follows [Ntime, Nmode] shape

        Returns:
            X: Numpy array for Input 
            Y: Numpy array for Output
        """

        from tqdm import tqdm 
        import numpy as np 

        if len(data.shape) <=2:
            data    = np.expand_dims(data,0)
        seqLen      = in_dim
        nSamples    = (data.shape[1]-seqLen)
        X           = np.empty([nSamples, seqLen, data.shape[-1]])
        Y           = np.empty([nSamples, next_step,data.shape[-1]])
        # Fill the input and output arrays with data
        k = 0
        for i in tqdm(np.arange(data.shape[0])):
            for j in np.arange(data.shape[1]-seqLen- next_step):
                X[k] = data[i, j        :j+seqLen]
                Y[k] = data[i, j+seqLen :j+seqLen+next_step]
                k    = k + 1
        print(f"The training data has been generated, has shape of {X.shape, Y.shape}")

        return X, Y
    
    def fitting(self, device,
        model,
        dl,
        loss_fn,
        Epoch,
        optimizer:torch.optim.Optimizer, 
        val_dl        = None,
        scheduler:list= None,
        if_early_stop = True,patience = 10,
        ):
    
        """
        A function for training loop

        Args: 
            device      :       the device for training, which should match the model
            
            model       :       The model to be trained
            
            dl          :       A dataloader for training
            
            loss_fn     :       Loss function
            
            Epochs      :       Number of epochs 
            
            optimizer   :       The optimizer object
            
            val_dl      :       The data for validation
            
            scheduler   :       A list of traning scheduler
            

        Returns:
            history: A dict contains training loss and validation loss (if have)

        """

        from tqdm import tqdm
        
        history = {}
        history["train_loss"] = []
        
        if val_dl:
            history["val_loss"] = []
        
        model.to(device)
        print(f"INFO: The model is assigned to device: {device} ")

        if scheduler is not None:
            print(f"INFO: The following schedulers are going to be used:")
            for sch in scheduler:
                print(f"{sch.__class__}")

        print(f"INFO: Training start")

        if if_early_stop: 
            early_stopper = EarlyStopper(patience=patience,min_delta=0)
            print("INFO: Early-Stopper prepared")

        for epoch in range(Epoch):
            #####
            #Training step
            #####
            model.train()
            loss_val = 0; num_batch = 0
            for batch in tqdm(dl):
                x, y = batch
                x = x.to(device).float(); y =y.to(device).float()
                optimizer.zero_grad()
                
                pred = model(x)
                loss = loss_fn(pred,y)
                loss.backward()
                optimizer.step()

                

                loss_val += loss.item()/x.shape[0]
                num_batch += 1

            history["train_loss"].append(loss_val/num_batch)

            if scheduler is not None:
                lr_now = 0 
                for sch in scheduler:
                    sch.step()
                    lr_now = sch.get_last_lr()
                print(f"INFO: Scheduler updated, LR = {lr_now} ")

            if val_dl:
            #####
            #Valdation step
            #####
                loss_val = 0 ; num_batch = 0 
                model.eval()
                for batch in (val_dl):
                    x, y = batch
                    x = x.to(device).float(); y =y.to(device).float()
                    pred = model(x)
                    loss = loss_fn(pred,y)
                
                    loss_val += loss.item()/x.shape[0]
                    num_batch += 1

                history["val_loss"].append(loss_val/num_batch)
            
            train_loss = history["train_loss"][-1]
            if val_dl:
                val_loss = history["val_loss"][-1]
            else:
                val_loss = None
            print(
                    f"At Epoch    = {epoch},\n"+\
                    f"Train_loss  = {train_loss}\n"+\
                    f"Val_loss    = {val_loss}\n"          
                )
            
            if if_early_stop:
                if early_stopper.early_stop(loss_val/num_batch):
                    print("Early-stopp Triggered, Going to stop the training")
                    break
        return history
    
    
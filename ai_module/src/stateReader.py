
from os.path import exists
import torch

class stateReader():
    def __init__(self):
        print("reader")

    def readState(self, statefile, lr, model, optimizer):
        
        if exists(statefile):
            state = torch.load(statefile)
            model.load_state_dict(state['model_state'])
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
            optimizer.load_state_dict(state['optim_state'])
            L = state['train_loss']
            V = state['validation_loss']
            return (L,V)
        
    def saveState(self, L, V, out_state_path, model, optimizer):

        state = {'model_state': model.module.state_dict() ,
         'optim_state': optimizer.state_dict(),
         'train_loss': L,
         'validation_loss':V}
        
        torch.save(state,out_state_path)

    def savePredictState(self, output, predict_output_file):
        torch.save(output,predict_output_file)
        
        print(f"INFO: The checkpoints has been saved!")
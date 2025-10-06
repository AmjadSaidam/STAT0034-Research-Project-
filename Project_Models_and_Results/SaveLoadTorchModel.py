# save pytorch model 
import torch 
import torch.nn as nn
from io import BytesIO as IO_BytesIO  # Rename to avoid conflicts

class SaveLoadPyTorchModel(nn.Module): 
    def __init__(self, model: nn.Module): # so we can access attributes in class methods 
        super().__init__()
        self.model = model

    def save_model(self):
        """Save Full Model 
        """
        buffer = IO_BytesIO()
        torch.save(self.model, buffer)
        buffer.seek(0)
        
        return buffer 

    def load_model(self, path: str): 
        """Load fully Trained model
        """
        self.model = torch.load(path, weights_only = False)
        
        return self.model

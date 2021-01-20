#!/usr/bin/env python3

import torch

class ModelComparator:
    
    def __init__(self, model_path1, model_path2):
        mod_arr1 = torch.load(model_path1)
        mod_arr2 = torch.load(model_path2)
        
        for i, (before, after) in enumerate(zip(mod_arr1, mod_arr2)):
            print(f"Loop {i}: {self.compare_model_parameters(before,after)}")
        
    def compare_model_parameters(self, model, other):
        for parms1, parms_other in zip(model.parameters(), other.parameters()):
            if parms1.data.ne(parms_other.data).sum() > 0:
                return False
        return True        

# ----------------- Main -------------
if __name__ == '__main__':
    ModelComparator('/home/paepcke/tmp/before_models.pth',
                    '/home/paepcke/tmp/after_models.pth')
            

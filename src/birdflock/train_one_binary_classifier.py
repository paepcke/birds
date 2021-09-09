'''
Created on Sep 4, 2021

@author: paepcke
'''

from skorch.classifier import NeuralNetBinaryClassifier
from skorch.dataset import CVSplit
from torchvision import transforms

from torch import cuda
from torch.nn import BCEWithLogitsLoss

from birdsong.binary_dataset import BinaryDataset
from birdsong.nets import NetUtils


class BinaryClassificationTrainer:
    '''
    classdocs
    '''

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, 
                 species_dirs_root, 
                 target_species,
                 transforms=None):
        '''
        The species_dirs_root contains one
        subdirectory for each species. The target_species
        is the name of the subdirectory whose recordings
        are examples of the binary classifier's target.
        The other subdirectories contain negative example
        recordings.
        
        transforms are instances of signal processing
        filters to apply.
        
        :param species_dirs_root: root of species-specific
            subdirectories
        :type species_dirs_root: str
        :param target_species: name of species to recognize
        :type target_species: str
        :param transforms: list of filters to apply to
            every file before usage
        :type trnasfo: {None | [Filter]
        '''
        
        self.num_classes  = 1
        self.pretrained   = True
        self.freeze       = False
        self.to_grayscale = True
        self.net_name     = 'resnet18'
        
        transforms = self.create_transforms()
        
        self.model    = NetUtils.get_net(self.net_name,
                                 num_classes=self.num_classes,
                                 pretrained=self.pretrained,
                                 freeze=self.freeze,
                                 to_grayscale=self.to_grayscale
                                 )
        
        cv_split = CVSplit()
        dataset  = BinaryDataset(species_dirs_root, target_species, transforms)

        # Use all defaults for optimizer, loss func, etc.
        # Maybe modify:
        #    o device <--- 'cuda'
        #****************
        #net = NeuralNet(self.model, criterion=torch.nn.NLLLoss,)
        net = NeuralNetBinaryClassifier(self.model,
                                        train_split=cv_split,
                                        #criterion=nn.CrossEntropyLoss,
                                        criterion=BCEWithLogitsLoss,
                                        dataset=dataset,
                                        device='cuda' if cuda.is_available() else 'cpu'
                                        )
        #****************
        #***********
        #for X,y in dataset:
        #    print(f"X: {X}; y: {y}")
        #***********
        net.fit(dataset, y=None)

        
    #------------------------------------
    # create_transforms
    #-------------------
    
    def create_transforms(self, 
                          to_grayscale=True,
                          sample_width=224,
                          sample_height=224
                          ):
        
        img_transforms = [
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                          transforms.Resize((sample_width, sample_height))
                          ]

        if to_grayscale:
            img_transforms.append(transforms.Grayscale())
                          
        return transforms.Compose(img_transforms)

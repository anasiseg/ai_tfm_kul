import torch
from torchvision import datasets
from torchvision import transforms
import numpy as np
from src.database.database_manager import DatabaseManager

class PILDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, augmentations):
        super(PILDataset, self).__init__()
        self.img_list = img_list
        self.augmentations = augmentations

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        return (self.augmentations(img),0)

class datasetManager():
    def __init__(self):
        print("dataset manager")
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float32)])#, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def preparing_data(self, data_path, trainpctg):
        # Transforms images to a PyTorch Tensor
        tensor_transform = transforms.Compose([transforms.ToTensor()])
        # Dataset
        all_data = datasets.ImageFolder(data_path, transform=tensor_transform)
        print("The total number of data available is", len(all_data))
        dataset=all_data
        # dataset  = torch.utils.data.Subset(all_data, np.random.choice(len(all_data), 20000, replace=False))
        trainset, testset = torch.utils.data.random_split(dataset, [int(trainpctg*len(dataset)), len(dataset)-int(trainpctg*len(dataset))])

        return (trainset, testset)
    

    def loading_sharp_data(self, trainpctg): 
        ddManager = DatabaseManager()
        ids=ddManager.get_all_harp_id()
        dt_array = []
        for id in ids:
            img_array=ddManager.get_harp_image(id_sharp=id)
            dt_array.append(PILDataset(img_array, self.transform))

        # # Created using indices from 0 to train_size.
        # trainset = torch.utils.data.Subset(dataset, range(int(trainpctg*len(dataset))))

        # # Created using indices from train_size to train_size + test_size.
        # testset = torch.utils.data.Subset(dataset, range(int(trainpctg*len(dataset)), len(dataset)))
        # # trainset, testset = torch.utils.data.split(dataset, [int(trainpctg*len(dataset)), len(dataset)-int(trainpctg*len(dataset))])
        return dt_array
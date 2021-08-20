import os
import torch
from torch.utils.data import Dataset, DataLoader
from src.data.dataset import Detection
from src.models.SoDA import SoDA

dataset_train = Detection(data_path=os.path.join("..","data","raw","MOT17","train"))

data_loader_train = DataLoader(dataset_train,batch_size=16,shuffle=True,num_workers=8)


m = SoDA()
detection = dataset_train[0][:,2:-1].unsqueeze(0)
z = m(detection)

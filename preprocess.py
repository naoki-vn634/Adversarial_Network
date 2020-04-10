import os
from glob import glob
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import pickle

def mnist_path_list(root_path='../data/img_78'):

    img_list_7 = [path for i, path in enumerate(glob(os.path.join(root_path,'img_7_*.jpg'))) if i < 200]
    img_list_8 = [path for i, path in enumerate(glob(os.path.join(root_path,'img_8_*.jpg'))) if i < 200]
    return img_list_7 + img_list_8

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class ImageTransfrom():
    def __init__(self,mean,std):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])

    def __call__(self,img):
        return self.data_transform(img)

class MNIST_IMG_Dataset(data.Dataset):

    def __init__(self,file_list,transform):
        self.file_list= file_list
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        
        return img_transformed
    
class CIFAR(object):

    def __init__(self,transform,root_path='../data/cifar-10-batches-py'):

        data = []
        label = []
        dirs = glob(os.path.join(root_path,'*/*'))
        for dir in dirs:
            datadict = unpickle(dir)
            data.append(datadict[b"data"])
            label.append(datadict[b"labels"])

        self.data = np.concatenate(data).reshape(-1,3,32,32)
        self.label = np.concatenate(label)
        self.transform = transform 
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        img, target = self.data[index], self.label[index]
        img=img.transpose(1,2,0)
        img = Image.fromarray(img)
        img_transformed = self.transform(img)

        return img_transformed
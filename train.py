import torch
from preprocess import *
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=64)
    args = parser.parse_args()

    # Define Dataset
    img_list = mnist_path_list()
    mean = (0.5,)
    std = (0.5,)
    train_dataset = MNIST_IMG_Dataset(img_list,transform=ImageTransfrom(mean,std))
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)

if __name__ == '__main__':
    main()
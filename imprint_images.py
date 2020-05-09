import os
import cv2
import torch
# import torchvison
import argparse

import numpy as np

from  PIL import Image
from glob import glob
from model import *


def imprint(image,file,out):
    count = 0
    for i,img in enumerate(image):
        #  Image.fromarray(np.uint8(np.asarray(image)))
        # scale = 255.0 / np.max(arrayimage)

        # pilImg = Image.fromarray(np.uint8(arrayimage*scale))
        img_np=img.cpu().detach().numpy()
        img_np = img_np.transpose(2,1,0)
        scale = 255.0/np.max(img_np)
        img_np = img_np * scale


        cv2.imwrite("result.png",img_np)
        
        if i % 4 ==0:
            img1 = img_np
        elif i % 4==3:
            img1 = np.hstack((img1,img_np))
            if count==0:
                img2 = img1
                count += 1
            else:
                img2 = np.vstack((img2,img1))
        else:
            img1 = np.hstack((img1,img_np))
            
    filename = file + '.png'
    
    cv2.imwrite(os.path.join(out,filename),img2)

                




def eval(net,device,weight_path,out):
    
    net.eval()
    for weight in weight_path:
        file_name = os.path.splitext(os.path.basename(weight))[0]
        net.load_state_dict(torch.load(weight))
        net.to(device)

        batchsize = 16 #表示する画像枚数
        z_dim = 20

        input_z = torch.randn(batchsize,z_dim,1,1).to(device)

        output = net(input_z)

        imprint(output,file_name,out)


        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_dir',type=str, default='/home/matsunaga/data/GAN/weight')
    parser.add_argument('--out_dir', type=str, default='/home/matsunaga/data/GAN')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print('device: ',device)

    G = DCGenerator(num_channel=1,image_size=64)

    weight_path = glob(os.path.join(args.weight_dir,'*.pth'))
    
    eval(net=G,device=device,weight_path=weight_path,out=args.out_dir)

if __name__ == '__main__':
    main()

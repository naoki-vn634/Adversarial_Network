import os
import torch
import torch.nn as nn
from preprocess import *
from model import *
import argparse
import time

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0)

def train(G, D, dataloader, output, num_epochs,interval):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print('device: ',device)

    g_lr, d_lr = 0.0001, 0.0004
    beta1, beta2 = 0.0, 0.9
    g_optimizer = torch.optim.Adam(G.parameters(), g_lr, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), d_lr, [beta1, beta2])

    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    
    z_dim = 20
    mini_batch_size = 64

    G = G.to(device)
    D = D.to(device)
    G.train()  
    D.train()  
    # torch.backends.cudnn.benchmark = True

    # 画像の枚数
    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    # イテレーションカウンタをセット
    iteration = 1
    logs = []

    # epochのループ
    for epoch in range(num_epochs):

        # 開始時刻を保存
        t_epoch_start = time.time()
        epoch_g_loss = 0.0  # epochの損失和
        epoch_d_loss = 0.0  # epochの損失和

        print('-------------')
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-------------')
        print('（train）')

        # データローダーからminibatchずつ取り出すループ
        for imges in dataloader:

            # --------------------
            # 1. Discriminatorの学習
            # --------------------
            # ミニバッチがサイズが1だと、バッチノーマライゼーションでエラーになるのでさける
            if imges.size()[0] == 1:
                continue

            # 正解ラベルと偽ラベルを作成
 
            mini_batch_size = imges.size()[0]
            label_real = torch.full((mini_batch_size,), 1).to(device)
            label_fake = torch.full((mini_batch_size,), 0).to(device)

            # 真の画像を判定
            d_out_real = D(imges.to(device))

            # 偽の画像を生成して判定
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = G(input_z)
            d_out_fake = D(fake_images)

            # 誤差を計算
            d_loss_real = criterion(d_out_real.view(-1), label_real)
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake

            # バックプロパゲーション
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            d_loss.backward()
            d_optimizer.step()

            # --------------------
            # 2. Generatorの学習
            # --------------------
            # 偽の画像を生成して判定
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = G(input_z)
            d_out_fake = D(fake_images)

            # 誤差を計算
            g_loss = criterion(d_out_fake.view(-1), label_real)

            # バックプロパゲーション
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # --------------------
            # 3. 記録
            # --------------------
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            iteration += 1

        # epochのphaseごとのlossと正解率
        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_D_Loss:{:.4f} ||Epoch_G_Loss:{:.4f}'.format(
            epoch, epoch_d_loss/batch_size, epoch_g_loss/batch_size))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()
        
        if (epoch+1) % interval == 0:
            if not os.path.isdir(output):
                os.makedirs(output)
            torch.save(G.state_dict(),os.path.join(output,'epoch_{}_weight.pth'.format(epoch)))
            print('save weight')
    return G, D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--ckpt_interval',type=int, default=50)
    parser.add_argument('--out_dir', type=str, default='/home/matsunaga/data/GAN')
    args = parser.parse_args()

    # Define Dataset
    mean = (0.485,0.456,0.406)
    std = (0.229, 0.224, 0.225)
    # train_dataset = CIFAR(transform=ImageTransfrom(mean,std))
    # train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)



    img_list = mnist_path_list()
    mean = (0.5,)
    std = (0.5,)
    train_dataset = MNIST_IMG_Dataset(img_list,transform=ImageTransfrom(mean,std))
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)

    # Define model
    G = DCGenerator(num_chnannel=1, image_size=64)
    D = DCDiscriminator(num_channel=1, image_size=64)
    G.apply(weight_init)
    D.apply(weight_init)
    print('weight and bias was initialized.')

    G_new,D_new = train(G=G,D=D,dataloader=train_dataloader, output=args.out_dir, num_epochs=args.epochs,interval=args.ckpt_interval)
    


if __name__ == '__main__':
    main()
import os
from Dataloader.model_net_cross_val import get_sets
import torch
from torchvision.utils import save_image
from util.pcview import PCViews
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def get_img(inpt):
    bs=inpt.shape[0]
    imgs=PCViews().get_img(inpt.permute(0,2,1))
        
    _,h,w=imgs.shape
        
    imgs=imgs.reshape(bs,6,-1)
    max=torch.max(imgs,-1,keepdim=True)[0]
    min=torch.min(imgs,-1,keepdim=True)[0]
        
    nor_img=(imgs-min)/(max-min+0.0001)
    nor_img=nor_img.reshape(bs,6,h,w)
    return nor_img
    
def img(inpt,type,num):
        
    norm_img=get_img(inpt) # (20,6,128,128)
    #norm_img.save()
    norm_img=norm_img.unsqueeze(2)

    root = "projimgSC/"
    for i  in range(norm_img.shape[0]):
        for j in range(6):
            path = os.path.join(root,type)
            try:
                os.mkdir(path)
            except:
                pass
            path = os.path.join(path,str(num))
            try:
                os.mkdir(path)
            except:
                pass
            path = os.path.join(path,str(i))
            try:
                os.mkdir(path)
            except:
                pass
            picname = str(j) + '.png'
            path = os.path.join(path, picname)
            save_image(norm_img[i, j, 0],path)


def main():
    path = "scanobjectnn_fs_cross/Data/"
    train_loader,val_loader=get_sets(data_path=path,fold=0,k_way=5,n_shot=1,query_num=10,data_aug=True)
    for i, (x_cpu,y_cpu) in enumerate(train_loader):
        x,y=x_cpu.to('cuda'),y_cpu.to('cuda')
        img(x,"train",i)
    for i, (x_cpu,y_cpu) in enumerate(train_loader):
        x,y=x_cpu.to('cuda'),y_cpu.to('cuda')
        img(x,"val",i)



if __name__=='__main__':
    main()
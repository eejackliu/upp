import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torchvision as tv
import random
import math
from functools import reduce
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader as dataloader
# import nonechucks as nc
from voc_seg import my_data,label_acc_score,voc_colormap,seg_target
vgg=tv.models.vgg11_bn(pretrained=True)
image_transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
mask_transform=transforms.Compose([transforms.ToTensor()])
trainset=my_data(transform=image_transform,target_transform=mask_transform)
testset=my_data(image_set='test',transform=image_transform,target_transform=mask_transform)
trainload=torch.utils.data.DataLoader(trainset,batch_size=1)
testload=torch.utils.data.DataLoader(testset,batch_size=1)
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device=torch.device('cpu')
print (device)

dtype=torch.float32
class deconv(nn.Module):
    def __init__(self,inchannel,middlechannel,outchannel,transpose=False):
        super(deconv,self).__init__()
        if transpose:
            self.block=nn.Sequential(nn.Conv2d(inchannel,middlechannel,3,padding=1),
                                   nn.BatchNorm2d(middlechannel),
                                   nn.ReLU(inplace=True),
                                   # nn.Conv2d(middlechannel,middlechannel,3,padding=1),
                                   # nn.BatchNorm2d(middlechannel),
                                   # nn.ReLU(inplace=True),
                                   # nn.ConvTranspose2d(middlechannel,outchannel,3,2), # use out_pading to minus one of padding
                                    nn.ConvTranspose2d(middlechannel,outchannel,3,2,1,1) # use out_pading to minus one of padding

                                     )
        else:
            self.block=nn.Sequential(nn.Conv2d(inchannel,middlechannel,3,padding=1),
                                   nn.BatchNorm2d(middlechannel),
                                   nn.ReLU(inplace=True),
                                   # nn.Conv2d(middlechannel,middlechannel,3,padding=0),
                                   # nn.BatchNorm2d(middlechannel),
                                   # nn.ReLU(inplace=True),
                                   nn.Conv2d(middlechannel,outchannel,1),         #since unsampling cann't change the channel num ,have to change channel num before next block
                                   nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True) # transpose is upsample and conv, now try conv and upsample need to confirm
                                     )
    def forward(self, input):
        return self.block(input)
class up(nn.Module):
    def __init__(self,inchannel_low,inchannel_same,middlechannel,outchannel,transpose=False):
        super(up,self).__init__()
        if  transpose:
            self.block=nn.ConvTranspose2d(inchannel_low,middlechannel,3,2,1,1)
            self.conv=nn.Sequential(nn.Conv2d(middlechannel+inchannel_same,outchannel,3,padding=1),
                                nn.BatchNorm2d(outchannel),
                                nn.ReLU(inplace=True),
                                # nn.ConvTranspose2d(middlechannel,outchannel,3,2,1,1)
                                )
        else:
            self.block=nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
                                     nn.Conv2d(inchannel_low,middlechannel,3,padding=1),
                                     nn.BatchNorm2d(middlechannel),
                                     nn.ReLU(inplace=True),)
            self.conv=nn.Sequential(
                                nn.Conv2d(middlechannel+inchannel_same,outchannel,3,padding=1),
                                nn.BatchNorm2d(outchannel),
                                nn.ReLU(inplace=True)
                                # nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
                              )

    def forward(self, uplayer,samlayer):
        self.up=self.block(uplayer)
        self.middle=torch.cat((self.up,samlayer),dim=1)
        self.out=self.conv(self.middle)
        return self.out
class conv(nn.Module):
    def __init__(self,inchannel,middlechannel,outchannel):
        super(conv,self).__init__()
        self.l1=nn.Sequential(nn.Conv2d(inchannel,middlechannel,3,padding=1),
                              nn.BatchNorm2d(middlechannel),
                              nn.Conv2d(middlechannel,outchannel,3,padding=1),
                              nn.BatchNorm2d(outchannel))
    def forward(self, input):
        return  self.l1(input)
class UPP(nn.Module):
    def __init__(self):
        super(UPP,self).__init__()
        self.l0_0=conv(3,32,32)
        self.l1_0=conv(32,64,64)
        self.l2_0=conv(64,128,128)
        self.l3_0=conv(128,256,256)
        self.l4_0=conv(256,512,512)
        self.l3_1=conv(512+256,512,256)
        self.l2_1=conv(256+128,256,128)
        self.l2_2=conv(128+128+256,256,128)
        self.l1_1=conv(128+64,128,64)
        self.l1_2=conv(128+64+64,128,64)
        self.l1_3=conv(128+64+64+64,128,64)
        self.l0_1=conv(64+32,64,32)
        self.l0_2=conv(64+32+32,64,32)
        self.l0_3=conv(64+32+32+32,64,32)
        self.l0_4=conv(64+32+32+32+32,64,32)
        self.pool=nn.MaxPool2d(2)
        self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.f1 =nn.Conv2d(32,1,1)
        self.f2 =nn.Conv2d(32,1,1)
        self.f3 =nn.Conv2d(32,1,1)
        self.f4 =nn.Conv2d(32,1,1)
    def forward(self, input):
        l0_0=self.l0_0(input)
        l1_0=self.l1_0(self.pool(l0_0))
        l2_0=self.l2_0(self.pool(l1_0))
        l3_0=self.l3_0(self.pool(l2_0))
        l4_0=self.l4_0(self.pool(l3_0))
        l3_1=self.l3_1(torch.cat((self.up(l4_0),l3_0),dim=1))
        l2_1=self.l2_1(torch.cat((self.up(l3_0),l2_0),dim=1))
        l2_2=self.l2_2(torch.cat((self.up(l3_1),l2_0,l2_1),dim=1))
        l1_1=self.l1_1(torch.cat((self.up(l2_0),l1_0),dim=1))
        l1_2=self.l1_2(torch.cat((self.up(l2_1),l1_0,l1_1),dim=1))
        l1_3=self.l1_3(torch.cat((self.up(l2_2),l1_0,l1_1,l1_2),dim=1))
        l0_1=self.l0_1(torch.cat((self.up(l1_0),l0_0),dim=1))
        l0_2=self.l0_2(torch.cat((self.up(l1_1),l0_0,l0_1),dim=1))
        l0_3=self.l0_3(torch.cat((self.up(l1_2),l0_0,l0_1,l0_2),dim=1))
        l0_4=self.l0_4(torch.cat((self.up(l1_3),l0_0,l0_1,l0_2,l0_3),dim=1))
        return self.f1(l0_1),self.f2(l0_2),self.f3(l0_3),self.f4(l0_4)



class Diceloss(nn.Module):
    def __init__(self):
        super(Diceloss,self).__init__()
    def dice_coef(self,x,y):
        numerator=2*torch.sum(x*y)+0.0001
        denominator=torch.sum(x**2)+torch.sum(y**2)+0.0001
        return numerator/denominator
    def forward(self, x,y):
        return 1-self.dice_coef(x,y)
class Bce_Diceloss(nn.Module):
    def __init__(self,bce_rate=0.5):
        super(Bce_Diceloss,self).__init__()
        self.rate=bce_rate
    def dice_coef(self,x,y):
        numerator=2*torch.sum(x*y)+0.0001
        denominator=torch.sum(x**2)+torch.sum(y**2)+0.0001
        return numerator/denominator
    def forward(self, x,y):
        return (1-self.dice_coef(x,y))*(1-self.rate)+self.rate*torch.nn.functional.binary_cross_entropy(x,y)
# def train(epoch):
#     model=UNET()
#     model.train()
#     model.to(device)
#     criterion=Diceloss()
#     optimize=torch.optim.Adam(model.parameters(),lr=0.0001)
#     for i in range(epoch):
#         tmp=0
#         for image,mask in trainload:
#             image,mask=image.to(device,dtype=dtype),mask.to(device,dtype=dtype)
#             optimize.zero_grad()
#             pred=model(image)
#             loss=criterion(pred,mask)
#             loss.backward()
#             optimize.step()
#             tmp=loss.data
#             # print ("loss ",tmp)
#             # break
#         print ("{0} epoch ,loss is {1}".format(i,tmp))
#     return model

def train(epoch):
    model=UPP()
    model.train()
    model=model.to(device)
    criterion=Diceloss()
    optimize=torch.optim.Adam(model.parameters(),lr=0.0001)
    store_loss=[]
    for i in range(epoch):
        tmp=0
        for image,mask in trainload:
            image,mask=image.to(device,dtype=dtype),mask.to(device,dtype=dtype)
            optimize.zero_grad()
            l1,l2,l3,l4,l5=model(image)
            loss_list=list(map(lambda x,y:criterion(x,y),[l1,l2,l3,l4,l5],[mask]*5))
            tmp=reduce(lambda x,y:x+y,loss_list)
            loss=tmp/5
            loss.backward()
            optimize.step()
            tmp=loss.data
            print ("loss ",tmp)
            # break
        store_loss.append(tmp)
        print ("{0} epoch ,loss is {1}".format(i,tmp))
    return model,store_loss
def test(model):
    img=[]
    pred=[]
    mask=[]
    l1_list=[]
    l2_list=[]
    l3_list=[]
    l4_list=[]
    l5_list=[]
    with torch.no_grad():
        model.eval()
        model.to(device)
        for image,mask_img in testload:
            image=image.to(device,dtype=dtype)
            l1,l2,l3,l4,output=model(image)
            label=output.cpu()>0.5
            # l1_list.append((l1>0.5).to(torch.long))
            # l2_list.append((l2>0.5).to(torch.long))
            # l3_list.append((l3>0.5).to(torch.long))
            # l4_list.append((l4>0.5).to(torch.long))
            l1_list.append(l1)
            l2_list.append(l2)
            l3_list.append(l3)
            l4_list.append(l4)
            l5_list.append(output)




            pred.append(label.to(torch.long))
            img.append(image.cpu())
            mask.append(mask_img)
    return torch.cat(img),torch.cat(pred),torch.cat(mask),[l1_list,l2_list,l3_list,l4_list,l5_list]

def picture(img,pred,mask):
    # all must bu numpy object
    plt.figure()
    mean,std=np.array((0.485, 0.456, 0.406)),np.array((0.229, 0.224, 0.225))
    num=len(img)
    tmp=img.transpose(0,2,3,1)
    tmp=tmp*std+mean
    tmp=np.concatenate((tmp,pred,mask),axis=0)
    for i,j in enumerate(tmp,1):
        plt.subplot(3,num,i)
        plt.imshow(j)
    plt.show()
def torch_pic(img,pred,mask):
    img, pred, mask = img[:4], pred[:4].to(torch.long), mask[:4].to(torch.long)
    pred = pred.squeeze(dim=1)
    mask = mask.squeeze(dim=1)
    voc_colormap = [[0, 0, 0], [245, 222, 179]]
    voc_colormap = torch.from_numpy(np.array(voc_colormap))
    voc_colormap = voc_colormap.to(dtype)
    mean, std = np.array((0.485, 0.456, 0.406)), np.array((0.229, 0.224, 0.225))
    mean, std = torch.from_numpy(mean).to(dtype), torch.from_numpy(std).to(dtype)
    img = img.permute(0, 2, 3, 1)
    img = (img * std + mean)
    # pred=pred.permute(0,2,3,1)
    # mask=mask.permute(0,2,3,1)
    pred = voc_colormap[pred] / 255.0
    mask = voc_colormap[mask] / 255.0
    pred=pred.permute(0, 3, 1, 2)
    mask=mask.permute(0, 3, 1, 2)
    tmp = tv.utils.make_grid(torch.cat((img.permute(0,3,1,2), pred, mask)), nrow=4)
    plt.imshow(tmp.permute(1,2,0))
    plt.show()
def my_iou(label_pred,label_mask):
    iou=[]
    for i,j in zip(label_pred.to(torch.float),label_mask):
        iou.append((i*j).sum()/(i.sum()+j.sum()-(i*j).sum()))
    return iou
def train_unet(epoch):

    model=UNET()
    model.train()
    model=model.to(device)
    criterion=Diceloss()
    optimise=torch.optim.Adam(model.parameters(),1e-3)
    tmp=0
    for i in range(epoch):
        for img,mask in trainload:
            img,mask=img.to(device,dtype=dtype),mask.to(device,dtype=dtype)
            optimise.zero_grad()
            output=model(img)
            loss=criterion(output,mask)
            loss.backward()
            optimise.step()
            tmp=loss.data
            # print (loss.data)
        print ('num {0} loss {1}'.format(i,tmp))
    return model

# from torchviz import make_dot
# a=torch.zeros(1,3,320,240)
# m=U_plus()
# d=m(a)
# make_dot(d,params=dict(m.named_parameters()))

# model,loss_list=train(60)
# torch.save(model.state_dict(),'uplus')
# model=U_plus()
# model.load_state_dict(torch.load('uplus',map_location='cpu'))
# img,pred,mask,l=test(model)
# ap,iou,hist,tmp=label_acc_score(mask,pred,2)
# # iu=my_iou(pred,mask)
# torch_pic(img[0:4],pred[0:4].to(torch.long),mask[0:4].to(torch.long))

# a=torch.zeros(1,3,320,240)
# tmp=UNET()
# tmp(a)

#%%
def test_unet(model):
    img=[]
    pred=[]
    mask=[]

    with torch.no_grad():
        model.eval()
        model=model.to(device)
        for image,mask_img in testload:
            image=image.to(device,dtype=dtype)
            output=model(image)
            label=output.cpu()>0.5
            # l1_list.append((l1>0.5).to(torch.long))
            # l2_list.append((l2>0.5).to(torch.long))
            # l3_list.append((l3>0.5).to(torch.long))
            # l4_list.append((l4>0.5).to(torch.long))
            pred.append(label.to(torch.long))
            img.append(image.cpu())
            mask.append(mask_img)
    return torch.cat(img),torch.cat(pred),torch.cat(mask)
# model=train_unet(20)
# torch.save(model.state_dict(),'unet')
# model=UNET()
# model.load_state_dict(torch.load(model.state_dict(),'unet'))
# img,pred,mask=test_unet(model)
# https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
# https://github.com/ybabakhin/kaggle_salt_bes_phalanx/blob/master/bes/losses.py
# https://arxiv.org/pdf/1606.04797.pdf#pdfjs.action=download
# okular
#https://arxiv.org/abs/1505.02496
# l0   l1
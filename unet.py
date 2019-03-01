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
trainload=torch.utils.data.DataLoader(trainset,batch_size=8)
testload=torch.utils.data.DataLoader(testset,batch_size=1)
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device=torch.device('cpu')
print (device)

dtype=torch.float32
class conv(nn.Module):
    def __init__(self,inchannel,middlechannel,outchannel):
        super(conv,self).__init__()
        self.l1=nn.Sequential(nn.Conv2d(inchannel,middlechannel,3,padding=1),
                              nn.BatchNorm2d(middlechannel),
                              nn.Conv2d(middlechannel,outchannel,3,padding=1),
                              nn.BatchNorm2d(outchannel))
    def forward(self, input):
        return  self.l1(input)
class up(nn.Module):
    def __init__(self,inchannel_low,inchannel_same,middlechannel,outchannel,transpose=False):
        super(up,self).__init__()
        if  transpose:
            self.block=nn.ConvTranspose2d(inchannel_low,middlechannel,2,2)
            self.conv=nn.Sequential(nn.Conv2d(inchannel_same+middlechannel,outchannel,3,padding=1),
                                nn.BatchNorm2d(outchannel),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(outchannel,outchannel,3,padding=1),
                                nn.BatchNorm2d(outchannel),
                                nn.ReLU(inplace=True),
                                # nn.ConvTranspose2d(middlechannel,outchannel,3,2,1,1)
                                )
        # else:
            # self.block = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #                            nn.Conv2d(inchannel_low, middlechannel, 3, padding=1),
            #                            nn.BatchNorm2d(middlechannel),
            #                            nn.ReLU(inplace=True), )
            # self.conv=nn.Sequential(nn.Conv2d(inchannel_same+middlechannel,outchannel,3,padding=1),
            #                     nn.BatchNorm2d(outchannel),
            #                     nn.ReLU(inplace=True),
            #
            #                     # nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
            #                   )
    def forward(self, uplayer,samlayer): #the uplayer need to be cropped and upsample
        tmp=self.block(uplayer)  # if block is transpose then need crop or it needs  pad(self.middle,(0,1,0,0),mode='replicate')
        return self.conv(torch.cat((tmp,samlayer),dim=1))
class pad_up(nn.Module):
    def __init__(self,inchannel_low,inchannel_same,middlechannel,outchannel,transpose=False):
        super(pad_up,self).__init__()
        if  transpose:
            self.block=nn.ConvTranspose2d(inchannel_low,middlechannel,3,2,1,1)
            self.conv=nn.Sequential(nn.Conv2d(inchannel_same+middlechannel,outchannel,3,padding=1),
                                nn.BatchNorm2d(outchannel),
                                nn.ReLU(inplace=True),
                                # nn.ConvTranspose2d(middlechannel,outchannel,3,2,1,1)
                                )
        else:
            self.block = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                       nn.Conv2d(inchannel_low, middlechannel, 3, padding=1),
                                       nn.BatchNorm2d(middlechannel),
                                       nn.ReLU(inplace=True), )
            self.conv=nn.Sequential(nn.Conv2d(inchannel_same+middlechannel,outchannel,3,padding=1),
                                nn.BatchNorm2d(outchannel),
                                nn.ReLU(inplace=True),

                                # nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
                              )
    def forward(self, uplayer,samlayer): #the uplayer need to be cropped and upsample
        tmp=self.block(uplayer)  # if block is transpose then need crop or it needs  pad(self.middle,(0,1,0,0),mode='replicate')
        uplayer=torch.nn.functional.pad(tmp,[0,1,0,0],mode='replicate')
        return self.conv(torch.cat((uplayer,samlayer),dim=1))

class crop_up(nn.Module):
    def __init__(self, inchannel_low, inchannel_same, middlechannel, outchannel, transpose=False):
        super(crop_up, self).__init__()
        if transpose:
            self.block = nn.ConvTranspose2d(inchannel_low, middlechannel, 3, 2)
            self.conv = nn.Sequential(nn.Conv2d(inchannel_same + middlechannel, outchannel, 3, padding=1),
                                      nn.BatchNorm2d(outchannel),
                                      nn.ReLU(inplace=True),
                                      # nn.ConvTranspose2d(middlechannel,outchannel,3,2,1,1)
                                      )
        else:
            self.block = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                       nn.Conv2d(inchannel_low, middlechannel, 3, padding=1),
                                       nn.BatchNorm2d(middlechannel),
                                       nn.ReLU(inplace=True), )
            self.conv = nn.Sequential(nn.Conv2d(inchannel_same + middlechannel, outchannel, 3, padding=1),
                                      nn.BatchNorm2d(outchannel),
                                      nn.ReLU(inplace=True),

                                      # nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
                                      )

    def forward(self, uplayer, samlayer):  # the uplayer need to be cropped and upsample
        tmp = self.block(
            uplayer)  # if block is transpose then need crop or it needs  pad(self.middle,(0,1,0,0),mode='replicate')
        uplayer = self.center_crop(tmp, samlayer)
        return self.conv(torch.cat((uplayer, samlayer), dim=1))

    def center_crop(self, img, target):
        h, w = img.shape[-2:]
        th, tw = target.shape[-2:]
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return img[..., i:i + th, j:j + tw]
class UPP(nn.Module):
    def __init__(self):
        super(UPP,self).__init__()
        self.l0_0=conv(3,32,32)
        self.l1_0=conv(32,64,64)
        self.l2_0=conv(64,128,128)
        self.l3_0=conv(128,256,256)
        self.l4_0=conv(256,512,512)
        self.l3_1=up(512,256,256,256,True)
        self.l2_1=up(256,128,128,128,True)
        self.l2_2=up(256,128+128,128,128,True)
        self.l1_1=up(128,64,64,64,True)
        self.l1_2=up(128,64+64,64,64,True)
        self.l1_3=up(128,64+64+64,64,64,True)
        self.l0_1=up(64,32,32,32,True)
        self.l0_2=up(64,32+32,32,32,True)
        self.l0_3=up(64,32+32+32,32,32,True)
        self.l0_4=up(64,32+32+32+32,32,32,True)
        self.pool=nn.MaxPool2d(2)
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
        l3_1=self.l3_1(l4_0,l3_0)
        l2_1=self.l2_1(l3_0,l2_0)
        l2_2=self.l2_2(l3_1,torch.cat((l2_0,l2_1),dim=1))
        l1_1=self.l1_1(l2_0,l1_0)
        l1_2=self.l1_2(l2_1,torch.cat((l1_0,l1_1),dim=1))
        l1_3=self.l1_3(l2_2,torch.cat((l1_0,l1_1,l1_2),dim=1))
        l0_1=self.l0_1(l1_0,l0_0)
        l0_2=self.l0_2(l1_1,torch.cat((l0_0,l0_1),dim=1))
        l0_3=self.l0_3(l1_2,torch.cat((l0_0,l0_1,l0_2),dim=1))
        l0_4=self.l0_4(l1_3,torch.cat((l0_0,l0_1,l0_2,l0_3),dim=1))
        return torch.sigmoid(self.f1(l0_1)),\
               torch.sigmoid(self.f2(l0_2)),\
               torch.sigmoid(self.f3(l0_3)),\
               torch.sigmoid(self.f4(l0_4))

class USS(nn.Module):
    def __init__(self):
        super(USS,self).__init__()
        self.l0_0=conv(3,32,32)
        self.l1_0=conv(32,64,64)
        self.l2_0=conv(64,128,128)
        self.l3_0=conv(128,256,256)
        self.l4_0=conv(256,512,512)
        self.l3_1=up(512,256,128,128,True)
        self.l2_1=up(256,128,64,64,True)
        self.l2_2=up(128,128+64,64,64,True)
        self.l1_1=up(128,64,32,32,True)
        self.l1_2=up(64,64+32,32,32,True)
        self.l1_3=up(64,64+32+32,32,32,True)
        self.l0_1=up(64,32,16,1,True)
        self.l0_2=up(32,32+1,16,1,True)
        self.l0_3=up(32,32+1+1,16,1,True)
        self.l0_4=up(32,32+1+1+1,16,1,True)
        self.pool=nn.MaxPool2d(2)
        # self.f1 =nn.Conv2d(32,1,1)
        # self.f2 =nn.Conv2d(32,1,1)
        # self.f3 =nn.Conv2d(32,1,1)
        # self.f4 =nn.Conv2d(32,1,1)
    def forward(self, input):
        l0_0=self.l0_0(input)
        l1_0=self.l1_0(self.pool(l0_0))
        l2_0=self.l2_0(self.pool(l1_0))
        l3_0=self.l3_0(self.pool(l2_0))
        l4_0=self.l4_0(self.pool(l3_0))
        l3_1=self.l3_1(l4_0,l3_0)
        l2_1=self.l2_1(l3_0,l2_0)
        l2_2=self.l2_2(l3_1,torch.cat((l2_0,l2_1),dim=1))
        l1_1=self.l1_1(l2_0,l1_0)
        l1_2=self.l1_2(l2_1,torch.cat((l1_0,l1_1),dim=1))
        l1_3=self.l1_3(l2_2,torch.cat((l1_0,l1_1,l1_2),dim=1))
        l0_1=self.l0_1(l1_0,l0_0)
        l0_2=self.l0_2(l1_1,torch.cat((l0_0,l0_1),dim=1))
        l0_3=self.l0_3(l1_2,torch.cat((l0_0,l0_1,l0_2),dim=1))
        l0_4=self.l0_4(l1_3,torch.cat((l0_0,l0_1,l0_2,l0_3),dim=1))
        # return torch.sigmoid(self.f1(l0_1)),\
        #        torch.sigmoid(self.f2(l0_2)),\
        #        torch.sigmoid(self.f3(l0_3)),\
        #        torch.sigmoid(self.f4(l0_4))
        return  l0_1,\
                l0_2,\
                l0_3,\
                l0_4
class UAA(nn.Module):
    def __init__(self):
        super(UAA,self).__init__()
        self.l0_0=conv(3,16,16)
        self.l1_0=conv(16,32,32)
        self.l2_0=conv(32,64,64)
        self.l3_0=conv(64,128,128)
        self.l4_0=conv(128,256,256)
        self.l3_1=up(256,128,64,64,True)
        self.l2_1=up(128,64,32,32,True)
        self.l2_2=up(64,64+32,32,32,True)
        self.l1_1=up(64,32,16,16,True)
        self.l1_2=up(32,32+16,16,16,True)
        self.l1_3=up(32,32+16+16,16,16,True)
        self.l0_1=up(32,16,8,1,True)
        self.l0_2=up(16,16+1,8,1,True)
        self.l0_3=up(16,16+1+1,8,1,True)
        self.l0_4=up(16,16+1+1+1,8,1,True)
        self.pool=nn.MaxPool2d(2)
        # self.f1 =nn.Conv2d(32,1,1)
        # self.f2 =nn.Conv2d(32,1,1)
        # self.f3 =nn.Conv2d(32,1,1)
        # self.f4 =nn.Conv2d(32,1,1)
    def forward(self, input):
        l0_0=self.l0_0(input)
        l1_0=self.l1_0(self.pool(l0_0))
        l2_0=self.l2_0(self.pool(l1_0))
        l3_0=self.l3_0(self.pool(l2_0))
        l4_0=self.l4_0(self.pool(l3_0))
        l3_1=self.l3_1(l4_0,l3_0)
        l2_1=self.l2_1(l3_0,l2_0)
        l2_2=self.l2_2(l3_1,torch.cat((l2_0,l2_1),dim=1))
        l1_1=self.l1_1(l2_0,l1_0)
        l1_2=self.l1_2(l2_1,torch.cat((l1_0,l1_1),dim=1))
        l1_3=self.l1_3(l2_2,torch.cat((l1_0,l1_1,l1_2),dim=1))
        l0_1=self.l0_1(l1_0,l0_0)
        l0_2=self.l0_2(l1_1,torch.cat((l0_0,l0_1),dim=1))
        l0_3=self.l0_3(l1_2,torch.cat((l0_0,l0_1,l0_2),dim=1))
        l0_4=self.l0_4(l1_3,torch.cat((l0_0,l0_1,l0_2,l0_3),dim=1))
        # return torch.sigmoid(self.f1(l0_1)),\
        #        torch.sigmoid(self.f2(l0_2)),\
        #        torch.sigmoid(self.f3(l0_3)),\
        #        torch.sigmoid(self.f4(l0_4))
        return  l0_1,\
                l0_2,\
                l0_3,\
                l0_4

class UWW(nn.Module):
    def __init__(self):
        super(UWW,self).__init__()
        self.l0_0=conv(3,16,16)
        self.l1_0=conv(16,32,32)
        self.l2_0=conv(32,64,64)
        self.l3_0=conv(64,128,128)
        self.l4_0=conv(128,256,256)
        self.middle=conv(256,512,512)
        self.l4_1=crop_up(512,256,128,128,True)
        self.l3_1=up(256,128,64,64,True)
        self.l3_2=up(128,128+64,64,64,True)
        self.l2_1=up(128,64,32,32,True)
        self.l2_2=up(64,64+32,32,32,True)
        self.l2_3=up(64,64+32+32,32,32,True)
        self.l1_1=up(64,32,16,16,True)
        self.l1_2=up(32,32+16,16,16,True)
        self.l1_3=up(32,32+16+16,16,16,True)
        self.l1_4=up(32,32+16+16+16,16,16,True)
        self.l0_1=up(32,16,8,1,True)
        self.l0_2=up(16,16+1,8,1,True)
        self.l0_3=up(16,16+1+1,8,1,True)
        self.l0_4=up(16,16+1+1+1,8,1,True)
        self.l0_5=up(16,16+1+1+1+1,8,1,True)
        self.pool=nn.MaxPool2d(2)
        # self.f1 =nn.Conv2d(32,1,1)
        # self.f2 =nn.Conv2d(32,1,1)
        # self.f3 =nn.Conv2d(32,1,1)
        # self.f4 =nn.Conv2d(32,1,1)
    def forward(self, input):
        l0_0=self.l0_0(input)
        l1_0=self.l1_0(self.pool(l0_0))
        l2_0=self.l2_0(self.pool(l1_0))
        l3_0=self.l3_0(self.pool(l2_0))
        l4_0=self.l4_0(self.pool(l3_0))
        middle=self.middle(self.pool(l4_0))
        l4_1=self.l4_1(middle,l4_0)
        l3_1=self.l3_1(l4_0,l3_0)
        l3_2=self.l3_2(l4_1,torch.cat((l3_0,l3_1),dim=1))
        l2_1=self.l2_1(l3_0,l2_0)
        l2_2=self.l2_2(l3_1,torch.cat((l2_0,l2_1),dim=1))
        l2_3=self.l2_3(l3_2,torch.cat((l2_0,l2_1,l2_2),dim=1))
        l1_1=self.l1_1(l2_0,l1_0)
        l1_2=self.l1_2(l2_1,torch.cat((l1_0,l1_1),dim=1))
        l1_3=self.l1_3(l2_2,torch.cat((l1_0,l1_1,l1_2),dim=1))
        l1_4=self.l1_4(l2_3,torch.cat((l1_0,l1_1,l1_2,l1_3),dim=1))
        l0_1=self.l0_1(l1_0,l0_0)
        l0_2=self.l0_2(l1_1,torch.cat((l0_0,l0_1),dim=1))
        l0_3=self.l0_3(l1_2,torch.cat((l0_0,l0_1,l0_2),dim=1))
        l0_4=self.l0_4(l1_3,torch.cat((l0_0,l0_1,l0_2,l0_3),dim=1))
        l0_5=self.l0_5(l1_4,torch.cat((l0_0,l0_1,l0_2,l0_3,l0_4),dim=1))
        # return torch.sigmoid(self.f1(l0_1)),\
        #        torch.sigmoid(self.f2(l0_2)),\
        #        torch.sigmoid(self.f3(l0_3)),\
        #        torch.sigmoid(self.f4(l0_4))
        return  l0_1,\
                l0_2,\
                l0_3,\
                l0_4,\
                l0_5
class Diceloss(nn.Module):
    def __init__(self):
        super(Diceloss,self).__init__()
    def dice_coef(self,x,y):
        numerator=2*torch.sum(x*y)+0.0001
        denominator=torch.sum(x**2)+torch.sum(y**2)+0.0001
        return numerator/denominator/float(len(y))
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
    optimize=torch.optim.Adam(model.parameters(),lr=0.001)
    store_loss=[]
    for i in range(epoch):
        tmp=0
        for image,mask in trainload:
            image,mask=image.to(device,dtype=dtype),mask.to(device,dtype=dtype)
            optimize.zero_grad()
            l1,l2,l3,l4=model(image)
            loss_list=list(map(lambda x,y:criterion(x,y),[l1,l2,l3,l4,],[mask]*4))
            tmp=reduce(lambda x,y:x+y,loss_list)
            loss=tmp/4
            loss.backward()
            optimize.step()
            tmp=loss.data
            # print ("loss ",tmp)
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
            l1,l2,l3,l4,l5=model(image)
            label=l5.cpu()>0.5
            # l1_list.append((l1>0.5).to(torch.long))
            # l2_list.append((l2>0.5).to(torch.long))
            # l3_list.append((l3>0.5).to(torch.long))
            # l4_list.append((l4>0.5).to(torch.long))
            l1_list.append(l1)
            l2_list.append(l2)
            l3_list.append(l3)
            l4_list.append(l4)
            l5_list.append(l5)



            pred.append(label.to(torch.long))
            img.append(image.cpu())
            mask.append(mask_img)
    return torch.cat(img),torch.cat(pred),torch.cat(mask),[l1_list,l2_list,l3_list,l4_list]

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

#upp,uss,uww,also need to ad drop=0.2

# torch.save(model.state_dict(),'uss')
model=UWW()
# model.load_state_dict(torch.load('uww'))

model.train()
model=model.to(device)
criterion=Diceloss()
optimize=torch.optim.Adam(model.parameters(),lr=0.001)
store_loss=[]
for i in range(20):
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
        # print ("loss ",tmp)
        # break
    store_loss.append(tmp)
    print ("{0} epoch ,loss is {1}".format(i,tmp))
torch.save(model.state_dict(),'uww')
# model=UPP()
# model.load_state_dict(torch.load('uplus',map_location='cpu'))
img,pred,mask,l=test(model)
ap,iou,hist,tmp=label_acc_score(mask,pred,2)
# # iu=my_iou(pred,mask)
# torch_pic(img[0:4],pred[0:4].to(torch.long),mask[0:4].to(torch.long))

# a=torch.zeros(1,3,320,240)
# tmp=UNET()
# tmp(a)

#%%

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